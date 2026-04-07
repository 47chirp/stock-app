from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco
from scipy import stats
import streamlit as st
import yfinance as yf


TRADING_DAYS = 252
MIN_TICKERS = 3
MAX_TICKERS = 10
BENCHMARK_TICKER = "^GSPC"
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_TICKERS = "AAPL, MSFT, NVDA"
CUSTOM_WEIGHT_MAX = 100.0


st.set_page_config(
	page_title="Interactive Portfolio Analytics Application",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="collapsed",
)


st.markdown(
	"""
	<style>
	.block-container {
		padding-top: 1.2rem;
		padding-bottom: 2rem;
	}
	.metric-card {
		border: 1px solid rgba(128, 128, 128, 0.25);
		border-radius: 0.8rem;
		padding: 0.85rem 1rem;
		background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,248,251,0.96));
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
	}
	.small-note {
		color: #5b6472;
		font-size: 0.92rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@dataclass
class DataBundle:
	prices: pd.DataFrame
	returns: pd.DataFrame
	selected_tickers: list[str]
	benchmark: str
	start_date: pd.Timestamp
	end_date: pd.Timestamp
	risk_free_rate: float
	warnings: list[str]


def parse_tickers(raw_text: str) -> list[str]:
	tokens = re.split(r"[\s,;]+", raw_text.upper().strip())
	tickers: list[str] = []
	for token in tokens:
		cleaned = token.strip().upper()
		if not cleaned:
			continue
		if cleaned not in tickers:
			tickers.append(cleaned)
	return tickers


def format_number(value: float, digits: int = 4) -> str:
	if value is None or pd.isna(value):
		return "N/A"
	if np.isinf(value):
		return "inf"
	return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 2) -> str:
	if value is None or pd.isna(value):
		return "N/A"
	if np.isinf(value):
		return "inf"
	return f"{value * 100:.{digits}f}%"


def normalize_slider_weights(weights: dict[str, float]) -> tuple[pd.Series, float]:
	series = pd.Series(weights, dtype=float)
	total = float(series.sum())
	if total <= 0:
		normalized = pd.Series(1.0 / len(series), index=series.index)
		return normalized, 0.0
	normalized = series / total
	return normalized, total


@st.cache_data(show_spinner=False)
def download_adjusted_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
	try:
		download = yf.download(
			ticker,
			start=start.to_pydatetime(),
			end=(end + pd.Timedelta(days=1)).to_pydatetime(),
			auto_adjust=False,
			actions=False,
			progress=False,
			threads=False,
			group_by="column",
		)
	except Exception as exc:  # pragma: no cover - network dependent
		return pd.Series(dtype=float, name=ticker), f"{ticker}: download failed ({exc})"

	if download is None or download.empty:
		return pd.Series(dtype=float, name=ticker), f"{ticker}: no price data returned"

	if "Adj Close" not in download.columns:
		return pd.Series(dtype=float, name=ticker), f"{ticker}: adjusted close data unavailable"

	series = download["Adj Close"].copy()
	if isinstance(series, pd.DataFrame):
		if ticker in series.columns:
			series = series[ticker]
		else:
			series = series.iloc[:, 0]

	series = series.dropna()
	series.index = pd.to_datetime(series.index).tz_localize(None)
	series.name = ticker

	if len(series) < 40:
		return series, f"{ticker}: insufficient history within the selected range"

	return series, ""


def clean_downloaded_data(
	selected_tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, list[str], list[str]]:
	downloaded: dict[str, pd.Series] = {}
	issues: list[str] = []

	for ticker in selected_tickers + [BENCHMARK_TICKER]:
		series, issue = download_adjusted_close(ticker, start, end)
		if issue:
			issues.append(issue)
		if not series.empty:
			downloaded[ticker] = series

	fatal_issues = [
		issue
		for issue in issues
		if "no price data returned" in issue
		or "adjusted close data unavailable" in issue
		or "insufficient history" in issue
	]
	if fatal_issues:
		raise ValueError("Invalid or insufficient ticker data: " + "; ".join(fatal_issues))

	if BENCHMARK_TICKER not in downloaded:
		raise ValueError(f"Benchmark {BENCHMARK_TICKER} could not be downloaded.")

	prices = pd.concat(downloaded.values(), axis=1, join="outer").sort_index()
	prices = prices.loc[(prices.index >= start) & (prices.index <= end)]

	missing_fraction = prices.isna().mean().to_dict()
	dropped = [ticker for ticker in selected_tickers if missing_fraction.get(ticker, 1.0) > 0.05]
	retained = [ticker for ticker in selected_tickers if ticker not in dropped]

	if dropped:
		issues.append(
			"Dropped tickers with more than 5% missing values in the selected range: "
			+ ", ".join(dropped)
		)

	if len(retained) < MIN_TICKERS:
		raise ValueError(
			"After removing tickers with excessive missing data, fewer than 3 valid stocks remain. "
			f"Dropped tickers: {', '.join(dropped) if dropped else 'none'}"
		)

	keep_cols = retained + [BENCHMARK_TICKER]
	prices = prices[keep_cols].dropna(how="any")

	if prices.empty:
		raise ValueError(
			"The selected securities do not share enough overlapping history after cleaning."
		)

	missing_rows = len(pd.concat(downloaded.values(), axis=1, join="outer")) - len(prices)
	if missing_rows > 0:
		issues.append(
			f"Aligned the dataset to overlapping dates and removed {missing_rows} row(s) with missing values."
		)

	return prices, retained, issues


@st.cache_data(show_spinner=False)
def build_analysis_payload(
	prices: pd.DataFrame,
	returns: pd.DataFrame,
	selected_tickers: list[str],
	benchmark: str,
	risk_free_rate: float,
) -> dict[str, object]:
	stats_table = annualized_statistics_table(returns[selected_tickers + [benchmark]], risk_free_rate)
	asset_returns = returns[selected_tickers].dropna()
	benchmark_returns = returns[benchmark].dropna()
	cov_matrix = covariance_matrix(asset_returns)
	corr_matrix = asset_returns.corr()

	equal_weights = pd.Series(equal_weight_vector(len(selected_tickers)), index=selected_tickers)
	equal_metrics = portfolio_metrics(equal_weights.to_numpy(), asset_returns, risk_free_rate)
	equal_series = portfolio_daily_series(equal_weights.to_numpy(), asset_returns)

	gmv_weights, gmv_error = optimize_min_variance(asset_returns)
	if gmv_weights is not None:
		gmv_weights_series = pd.Series(gmv_weights, index=selected_tickers)
		gmv_metrics = portfolio_metrics(gmv_weights, asset_returns, risk_free_rate)
		gmv_series = portfolio_daily_series(gmv_weights, asset_returns)
		gmv_prc = pd.Series(risk_contribution(gmv_weights, cov_matrix.to_numpy()), index=selected_tickers)
	else:
		gmv_weights_series = None
		gmv_metrics = None
		gmv_series = None
		gmv_prc = None

	tangency_weights, tan_error = optimize_max_sharpe(asset_returns, risk_free_rate)
	if tangency_weights is not None:
		tangency_weights_series = pd.Series(tangency_weights, index=selected_tickers)
		tangency_metrics = portfolio_metrics(tangency_weights, asset_returns, risk_free_rate)
		tangency_series = portfolio_daily_series(tangency_weights, asset_returns)
		tangency_prc = pd.Series(risk_contribution(tangency_weights, cov_matrix.to_numpy()), index=selected_tickers)
	else:
		tangency_weights_series = None
		tangency_metrics = None
		tangency_series = None
		tangency_prc = None

	frontier, frontier_error = efficient_frontier(asset_returns, risk_free_rate)
	benchmark_metrics = portfolio_metrics_from_series(benchmark_returns, risk_free_rate)

	return {
		"stats_table": stats_table,
		"asset_returns": asset_returns,
		"benchmark_returns": benchmark_returns,
		"cov_matrix": cov_matrix,
		"corr_matrix": corr_matrix,
		"equal_weights": equal_weights,
		"equal_metrics": equal_metrics,
		"equal_series": equal_series,
		"gmv_weights_series": gmv_weights_series,
		"gmv_metrics": gmv_metrics,
		"gmv_series": gmv_series,
		"gmv_prc": gmv_prc,
		"gmv_error": gmv_error,
		"tangency_weights_series": tangency_weights_series,
		"tangency_metrics": tangency_metrics,
		"tangency_series": tangency_series,
		"tangency_prc": tangency_prc,
		"tan_error": tan_error,
		"frontier": frontier,
		"frontier_error": frontier_error,
		"benchmark_metrics": benchmark_metrics,
	}


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
	return prices.pct_change().dropna(how="any")


def annualized_return_from_returns(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
	return returns.mean() * TRADING_DAYS


def annualized_volatility_from_returns(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
	return returns.std(ddof=1) * math.sqrt(TRADING_DAYS)


def downside_deviation(returns: pd.Series, annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
	target_daily = annual_risk_free_rate / TRADING_DAYS
	downside = np.minimum(returns - target_daily, 0.0)
	downside_dev = np.sqrt(np.mean(np.square(downside))) * math.sqrt(TRADING_DAYS)
	return float(downside_dev)


def sharpe_ratio(returns: pd.Series, annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
	ann_return = float(annualized_return_from_returns(returns))
	ann_vol = float(annualized_volatility_from_returns(returns))
	if ann_vol == 0:
		return np.nan
	return (ann_return - annual_risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
	ann_return = float(annualized_return_from_returns(returns))
	dd = downside_deviation(returns, annual_risk_free_rate)
	if dd == 0:
		return np.inf if ann_return > annual_risk_free_rate else np.nan
	return (ann_return - annual_risk_free_rate) / dd


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Series]:
	wealth = (1.0 + returns.fillna(0.0)).cumprod()
	running_max = wealth.cummax()
	drawdown = wealth / running_max - 1.0
	return float(drawdown.min()), drawdown


@st.cache_data(show_spinner=False)
def summary_statistics(returns: pd.DataFrame, annual_risk_free_rate: float) -> pd.DataFrame:
	rows = []
	for column in returns.columns:
		series = returns[column].dropna()
		rows.append(
			{
				"Series": column,
				"Annualized Mean Return": float(annualized_return_from_returns(series)),
				"Annualized Volatility": float(annualized_volatility_from_returns(series)),
				"Skewness": float(stats.skew(series, bias=False)),
				"Kurtosis": float(stats.kurtosis(series, fisher=True, bias=False)),
				"Minimum Daily Return": float(series.min()),
				"Maximum Daily Return": float(series.max()),
				"Sharpe Ratio": float(sharpe_ratio(series, annual_risk_free_rate)),
				"Sortino Ratio": float(sortino_ratio(series, annual_risk_free_rate)),
			}
		)
	return pd.DataFrame(rows).set_index("Series")


def rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
	return returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
	return series_a.rolling(window).corr(series_b)


@st.cache_data(show_spinner=False)
def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
	return returns.cov()


def portfolio_return(weights: np.ndarray, mean_daily_returns: np.ndarray) -> float:
	return float(np.dot(weights, mean_daily_returns) * TRADING_DAYS)


def portfolio_volatility(weights: np.ndarray, cov_daily: np.ndarray) -> float:
	variance_daily = float(weights.T @ cov_daily @ weights)
	return float(np.sqrt(max(variance_daily, 0.0) * TRADING_DAYS))


def portfolio_sortino_ratio(
	portfolio_returns: pd.Series, annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> float:
	return sortino_ratio(portfolio_returns, annual_risk_free_rate)


def portfolio_metrics(
	weights: np.ndarray,
	returns: pd.DataFrame,
	annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, float]:
	mean_daily = returns.mean().to_numpy()
	cov_daily = returns.cov().to_numpy()
	portfolio_daily_returns = pd.Series(returns.to_numpy() @ weights, index=returns.index)

	ann_return = portfolio_return(weights, mean_daily)
	ann_vol = portfolio_volatility(weights, cov_daily)
	sharpe = np.nan if ann_vol == 0 else (ann_return - annual_risk_free_rate) / ann_vol
	sortino = portfolio_sortino_ratio(portfolio_daily_returns, annual_risk_free_rate)
	mdd, _ = max_drawdown(portfolio_daily_returns)

	return {
		"annual_return": ann_return,
		"annual_volatility": ann_vol,
		"sharpe_ratio": float(sharpe),
		"sortino_ratio": float(sortino),
		"max_drawdown": mdd,
	}


@st.cache_data(show_spinner=False)
def portfolio_metrics_from_series(
	returns: pd.Series, annual_risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> dict[str, float]:
	ann_return = float(annualized_return_from_returns(returns))
	ann_vol = float(annualized_volatility_from_returns(returns))
	sharpe = np.nan if ann_vol == 0 else (ann_return - annual_risk_free_rate) / ann_vol
	sortino = float(sortino_ratio(returns, annual_risk_free_rate))
	mdd, _ = max_drawdown(returns)
	return {
		"annual_return": ann_return,
		"annual_volatility": ann_vol,
		"sharpe_ratio": float(sharpe),
		"sortino_ratio": sortino,
		"max_drawdown": mdd,
	}


def risk_contribution(weights: np.ndarray, cov_daily: np.ndarray) -> np.ndarray:
	portfolio_variance = float(weights.T @ cov_daily @ weights)
	if portfolio_variance <= 0:
		return np.full_like(weights, np.nan, dtype=float)
	marginal = cov_daily @ weights
	contributions = weights * marginal / portfolio_variance
	return contributions


def equal_weight_vector(n_assets: int) -> np.ndarray:
	return np.repeat(1.0 / n_assets, n_assets)


@st.cache_data(show_spinner=False)
def optimize_min_variance(returns: pd.DataFrame) -> tuple[np.ndarray | None, str]:
	n_assets = returns.shape[1]
	cov_daily = returns.cov().to_numpy()
	x0 = equal_weight_vector(n_assets)

	constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
	bounds = tuple((0.0, 1.0) for _ in range(n_assets))

	result = sco.minimize(
		lambda w: float(w.T @ cov_daily @ w),
		x0=x0,
		method="SLSQP",
		bounds=bounds,
		constraints=constraints,
	)

	if not result.success:
		return None, f"Global Minimum Variance optimization failed: {result.message}"
	return np.asarray(result.x, dtype=float), ""


@st.cache_data(show_spinner=False)
def optimize_max_sharpe(
	returns: pd.DataFrame, annual_risk_free_rate: float
) -> tuple[np.ndarray | None, str]:
	n_assets = returns.shape[1]
	cov_daily = returns.cov().to_numpy()
	mean_daily = returns.mean().to_numpy()
	x0 = equal_weight_vector(n_assets)

	constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
	bounds = tuple((0.0, 1.0) for _ in range(n_assets))

	def objective(weights: np.ndarray) -> float:
		annual_return = portfolio_return(weights, mean_daily)
		annual_vol = portfolio_volatility(weights, cov_daily)
		if annual_vol <= 0:
			return 1e6
		return -((annual_return - annual_risk_free_rate) / annual_vol)

	result = sco.minimize(
		objective,
		x0=x0,
		method="SLSQP",
		bounds=bounds,
		constraints=constraints,
	)

	if not result.success:
		return None, f"Tangency optimization failed: {result.message}"
	return np.asarray(result.x, dtype=float), ""


@st.cache_data(show_spinner=False)
def efficient_frontier(
	returns: pd.DataFrame,
	annual_risk_free_rate: float,
	n_points: int = 35,
) -> tuple[pd.DataFrame, str]:
	mean_daily = returns.mean().to_numpy()
	ann_means = mean_daily * TRADING_DAYS
	cov_daily = returns.cov().to_numpy()
	n_assets = len(ann_means)
	x0 = equal_weight_vector(n_assets)

	target_returns = np.linspace(float(ann_means.min()), float(ann_means.max()), n_points)
	frontier_rows: list[dict[str, float]] = []

	for target in target_returns:
		constraints = (
			{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
			{"type": "eq", "fun": lambda w, target=target: float(np.dot(w, ann_means) - target)},
		)
		bounds = tuple((0.0, 1.0) for _ in range(n_assets))

		result = sco.minimize(
			lambda w: float(w.T @ cov_daily @ w),
			x0=x0,
			method="SLSQP",
			bounds=bounds,
			constraints=constraints,
		)
		if not result.success:
			continue

		weights = np.asarray(result.x, dtype=float)
		annual_return = portfolio_return(weights, mean_daily)
		annual_vol = portfolio_volatility(weights, cov_daily)
		sharpe = np.nan if annual_vol == 0 else (annual_return - annual_risk_free_rate) / annual_vol
		frontier_rows.append(
			{
				"return": annual_return,
				"volatility": annual_vol,
				"sharpe": sharpe,
			}
		)
		x0 = weights

	if not frontier_rows:
		return pd.DataFrame(), "No feasible points could be generated for the efficient frontier."

	frontier = pd.DataFrame(frontier_rows).drop_duplicates().sort_values("volatility").reset_index(drop=True)
	return frontier, ""


def wealth_index(returns: pd.DataFrame | pd.Series, initial_value: float = 10000.0) -> pd.DataFrame | pd.Series:
	return initial_value * (1.0 + returns).cumprod()


def annualized_statistics_table(returns: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
	table = summary_statistics(returns, risk_free_rate)
	table.index.name = "Series"
	return table


def create_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
	fig = go.Figure(
		data=go.Heatmap(
			z=matrix.to_numpy(),
			x=matrix.columns,
			y=matrix.index,
			colorscale="RdBu",
			zmid=0,
			text=np.round(matrix.to_numpy(), 2),
			texttemplate="%{text}",
			textfont={"size": 11},
			colorbar={"title": "Value"},
		)
	)
	fig.update_layout(title=title, xaxis_title="Series", yaxis_title="Series", height=650)
	return fig


def create_cumulative_wealth_chart(prices: pd.DataFrame, selected_series: list[str]) -> go.Figure:
	returns = compute_daily_returns(prices[selected_series])
	wealth = wealth_index(returns)
	fig = go.Figure()
	for column in wealth.columns:
		fig.add_trace(go.Scatter(x=wealth.index, y=wealth[column], mode="lines", name=column))
	fig.update_layout(
		title="Cumulative Wealth from $10,000 Initial Investment",
		xaxis_title="Date",
		yaxis_title="Wealth Value ($)",
		legend_title="Series",
		height=550,
	)
	return fig


def create_distribution_chart(series: pd.Series) -> go.Figure:
	mu = float(series.mean())
	sigma = float(series.std(ddof=1))
	x_grid = np.linspace(series.min(), series.max(), 300)
	density = stats.norm.pdf(x_grid, mu, sigma) if sigma > 0 else np.zeros_like(x_grid)

	fig = go.Figure()
	fig.add_trace(
		go.Histogram(
			x=series,
			histnorm="probability density",
			name="Daily Returns",
			opacity=0.72,
			nbinsx=40,
		)
	)
	fig.add_trace(
		go.Scatter(
			x=x_grid,
			y=density,
			mode="lines",
			name="Fitted Normal",
			line={"width": 3},
		)
	)
	fig.update_layout(
		title=f"Histogram of Daily Returns with Fitted Normal Overlay: {series.name}",
		xaxis_title="Daily Simple Return",
		yaxis_title="Density",
		barmode="overlay",
		height=550,
	)
	return fig


def create_qq_plot(series: pd.Series) -> go.Figure:
	(theoretical, sample), (slope, intercept, _) = stats.probplot(series.dropna(), dist="norm")
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers", name="Observed Quantiles"))
	line_x = np.array([min(theoretical), max(theoretical)])
	line_y = slope * line_x + intercept
	fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="45° Reference", line={"dash": "dash"}))
	fig.update_layout(
		title=f"Q-Q Plot Against Theoretical Normal Distribution: {series.name}",
		xaxis_title="Theoretical Quantiles",
		yaxis_title="Sample Quantiles",
		height=550,
	)
	return fig


def create_rolling_volatility_chart(returns: pd.DataFrame, window: int) -> go.Figure:
	rolling = rolling_volatility(returns, window)
	fig = go.Figure()
	for column in rolling.columns:
		fig.add_trace(go.Scatter(x=rolling.index, y=rolling[column], mode="lines", name=column))
	fig.update_layout(
		title=f"Rolling Annualized Volatility ({window}-Day Window)",
		xaxis_title="Date",
		yaxis_title="Annualized Volatility",
		legend_title="Series",
		height=550,
	)
	return fig


def create_drawdown_chart(returns: pd.Series) -> tuple[go.Figure, float]:
	max_dd, drawdown = max_drawdown(returns)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name="Drawdown"))
	fig.update_layout(
		title=f"Drawdown Over Time: {returns.name}",
		xaxis_title="Date",
		yaxis_title="Drawdown",
		height=500,
	)
	return fig, max_dd


def create_rolling_correlation_chart(series_a: pd.Series, series_b: pd.Series, window: int) -> go.Figure:
	rolling_corr = rolling_correlation(series_a, series_b, window)
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{series_a.name} vs {series_b.name}")
	)
	fig.add_hline(y=0, line_dash="dash", line_color="gray")
	fig.update_layout(
		title=f"Rolling Correlation ({window}-Day Window): {series_a.name} vs {series_b.name}",
		xaxis_title="Date",
		yaxis_title="Correlation",
		height=500,
	)
	return fig


def create_weights_bar_chart(weights: pd.Series, title: str) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(go.Bar(x=weights.index, y=weights.values, marker_color="#1f77b4"))
	fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Weight", height=500)
	return fig


def create_risk_contribution_chart(contributions: pd.Series, title: str) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(go.Bar(x=contributions.index, y=contributions.values, marker_color="#ff7f0e"))
	fig.update_layout(title=title, xaxis_title="Asset", yaxis_title="Risk Contribution", height=500)
	return fig


def create_frontier_chart(
	frontier: pd.DataFrame,
	asset_metrics: pd.DataFrame,
	benchmark_point: dict[str, float],
	equal_point: dict[str, float],
	gmv_point: dict[str, float] | None,
	tangency_point: dict[str, float] | None,
	custom_point: dict[str, float] | None,
	annual_risk_free_rate: float,
) -> go.Figure:
	fig = go.Figure()

	if not frontier.empty:
		fig.add_trace(
			go.Scatter(
				x=frontier["volatility"],
				y=frontier["return"],
				mode="lines",
				name="Efficient Frontier",
				line={"width": 3, "color": "#1f77b4"},
			)
		)

	for asset in asset_metrics.index:
		fig.add_trace(
			go.Scatter(
				x=[asset_metrics.loc[asset, "Annualized Volatility"]],
				y=[asset_metrics.loc[asset, "Annualized Mean Return"]],
				mode="markers+text",
				name=asset,
				text=[asset],
				textposition="top center",
				marker={"size": 11},
			)
		)

	fig.add_trace(
		go.Scatter(
			x=[benchmark_point["volatility"]],
			y=[benchmark_point["return"]],
			mode="markers+text",
			name=BENCHMARK_TICKER,
			text=[BENCHMARK_TICKER],
			textposition="top center",
			marker={"size": 12, "symbol": "diamond", "color": "black"},
		)
	)

	fig.add_trace(
		go.Scatter(
			x=[equal_point["volatility"]],
			y=[equal_point["return"]],
			mode="markers+text",
			name="Equal Weight",
			text=["Equal Weight"],
			textposition="top center",
			marker={"size": 13, "symbol": "star", "color": "green"},
		)
	)

	if gmv_point is not None:
		fig.add_trace(
			go.Scatter(
				x=[gmv_point["volatility"]],
				y=[gmv_point["return"]],
				mode="markers+text",
				name="GMV",
				text=["GMV"],
				textposition="bottom center",
				marker={"size": 13, "symbol": "star-diamond", "color": "purple"},
			)
		)

	if tangency_point is not None:
		fig.add_trace(
			go.Scatter(
				x=[tangency_point["volatility"]],
				y=[tangency_point["return"]],
				mode="markers+text",
				name="Tangency",
				text=["Tangency"],
				textposition="bottom center",
				marker={"size": 13, "symbol": "star-square", "color": "crimson"},
			)
		)

		frontier_max_vol = max(frontier["volatility"].max() if not frontier.empty else 0.0, tangency_point["volatility"] * 1.2)
		cal_x = np.linspace(0, frontier_max_vol, 100)
		cal_y = annual_risk_free_rate + tangency_point["sharpe_ratio"] * cal_x
		fig.add_trace(
			go.Scatter(
				x=cal_x,
				y=cal_y,
				mode="lines",
				name="Capital Allocation Line",
				line={"dash": "dash", "color": "darkorange", "width": 3},
			)
		)

	if custom_point is not None:
		fig.add_trace(
			go.Scatter(
				x=[custom_point["volatility"]],
				y=[custom_point["return"]],
				mode="markers+text",
				name="Custom Portfolio",
				text=["Custom"],
				textposition="top center",
				marker={"size": 13, "symbol": "circle-open", "color": "#2ca02c", "line": {"width": 2}},
			)
		)

	fig.update_layout(
		title="Efficient Frontier, Asset Points, and Portfolio Overlays",
		xaxis_title="Annualized Volatility",
		yaxis_title="Annualized Return",
		height=650,
		legend_title="Series",
	)
	return fig


def build_portfolio_table(
	portfolios: dict[str, dict[str, float]],
	weights: dict[str, pd.Series],
) -> pd.DataFrame:
	rows = []
	for name, metrics in portfolios.items():
		row = {
			"Portfolio": name,
			"Annualized Return": metrics["annual_return"],
			"Annualized Volatility": metrics["annual_volatility"],
			"Sharpe Ratio": metrics["sharpe_ratio"],
			"Sortino Ratio": metrics["sortino_ratio"],
			"Maximum Drawdown": metrics["max_drawdown"],
		}
		row.update({f"Weight: {asset}": weight for asset, weight in weights[name].items()})
		rows.append(row)
	table = pd.DataFrame(rows).set_index("Portfolio")
	return table


def supported_lookback_windows(num_observations: int) -> dict[str, int]:
	windows = {"Trailing 1 Year": TRADING_DAYS, "Trailing 3 Years": TRADING_DAYS * 3, "Trailing 5 Years": TRADING_DAYS * 5}
	supported = {"Full Sample": num_observations}
	for label, days in windows.items():
		if num_observations >= days:
			supported[label] = days
	return supported


@st.cache_data(show_spinner=False)
def compute_window_portfolios(
	returns: pd.DataFrame, annual_risk_free_rate: float, window_days: int
) -> dict[str, object]:
	window_returns = returns.tail(window_days)
	gmv_weights, gmv_error = optimize_min_variance(window_returns)
	tangency_weights, tan_error = optimize_max_sharpe(window_returns, annual_risk_free_rate)
	return {
		"window_returns": window_returns,
		"gmv_weights": gmv_weights,
		"gmv_error": gmv_error,
		"tangency_weights": tangency_weights,
		"tan_error": tan_error,
	}


def portfolio_daily_series(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
	return pd.Series(returns.to_numpy() @ weights, index=returns.index)


def app_header() -> None:
	st.title("Interactive Portfolio Analytics Application")
	st.caption(
		"Analyze stocks, benchmark them against the S&P 500, and compare optimized portfolios using simple returns and no-short-selling constraints."
	)


def show_data_download_panel(bundle: DataBundle) -> None:
	with st.expander("Data Summary and Downloads", expanded=True):
		st.write(
			f"**Retained stocks:** {', '.join(bundle.selected_tickers)}  |  **Benchmark:** {bundle.benchmark}  |  **Observations:** {len(bundle.prices):,}"
		)
		st.write(f"**Analysis window:** {bundle.start_date.date()} to {bundle.end_date.date()}")
		st.download_button(
			"Download Cleaned Prices CSV",
			data=bundle.prices.to_csv().encode("utf-8"),
			file_name="cleaned_prices.csv",
			mime="text/csv",
		)
		st.download_button(
			"Download Daily Returns CSV",
			data=bundle.returns.to_csv().encode("utf-8"),
			file_name="daily_returns.csv",
			mime="text/csv",
		)


def main() -> None:
	app_header()

	if "bundle" not in st.session_state:
		st.session_state.bundle = None
	if "analysis_cache" not in st.session_state:
		st.session_state.analysis_cache = None

	tabs = st.tabs(
		[
			"Inputs / Data",
			"Exploratory Analysis",
			"Risk & Correlation",
			"Portfolio Optimization",
			"Estimation Window Sensitivity",
			"About / Methodology",
		]
	)

	with tabs[0]:
		st.subheader("Inputs / Data")
		st.write("Enter 3 to 10 tickers, choose a date range of at least two years, and fetch the data manually.")

		with st.form("data_form", clear_on_submit=False):
			ticker_text = st.text_area("Ticker symbols", value=DEFAULT_TICKERS, help="Use commas, spaces, or line breaks to separate tickers.")
			col1, col2, col3 = st.columns(3)
			with col1:
				default_start = date.today() - timedelta(days=365 * 5)
				start_date = st.date_input("Start date", value=default_start)
			with col2:
				end_date = st.date_input("End date", value=date.today())
			with col3:
				annual_rf = st.number_input(
					"Annualized risk-free rate",
					min_value=0.0,
					max_value=0.20,
					value=float(DEFAULT_RISK_FREE_RATE),
					step=0.005,
					format="%.3f",
				)

			fetch_clicked = st.form_submit_button("Fetch Market Data")

		if fetch_clicked:
			tickers = parse_tickers(ticker_text)
			if BENCHMARK_TICKER in tickers:
				tickers = [ticker for ticker in tickers if ticker != BENCHMARK_TICKER]
				st.warning(f"{BENCHMARK_TICKER} is reserved as the benchmark and was removed from the input ticker list.")
			if len(tickers) < MIN_TICKERS or len(tickers) > MAX_TICKERS:
				st.error("Please enter between 3 and 10 unique ticker symbols.")
			elif pd.Timestamp(end_date) <= pd.Timestamp(start_date):
				st.error("End date must be after start date.")
			elif pd.Timestamp(end_date) - pd.Timestamp(start_date) < pd.Timedelta(days=730):
				st.error("The selected date range must be at least 2 years.")
			else:
				with st.spinner("Downloading adjusted close prices and cleaning the dataset..."):
					try:
						prices, retained_tickers, warnings_list = clean_downloaded_data(
							tickers, pd.Timestamp(start_date), pd.Timestamp(end_date)
						)
						returns = compute_daily_returns(prices)
						if len(returns) < 40:
							raise ValueError("Not enough overlapping return observations after cleaning.")

						bundle = DataBundle(
							prices=prices,
							returns=returns,
							selected_tickers=retained_tickers,
							benchmark=BENCHMARK_TICKER,
							start_date=pd.Timestamp(start_date),
							end_date=pd.Timestamp(end_date),
							risk_free_rate=float(annual_rf),
							warnings=warnings_list,
						)
						st.session_state.bundle = bundle
						st.session_state.analysis_cache = None
						st.success("Market data loaded successfully.")
					except Exception as exc:
						st.error(str(exc))

		bundle: DataBundle | None = st.session_state.bundle
		if bundle is None:
			st.info("Fetch data to unlock the remaining analysis tabs.")
		else:
			show_data_download_panel(bundle)
			if bundle.warnings:
				for warning in bundle.warnings:
					st.warning(warning)

	bundle = st.session_state.bundle
	if bundle is None:
		for tab in tabs[1:]:
			with tab:
				st.info("Load data from the Inputs / Data tab to view this section.")
		with tabs[-1]:
			st.markdown(
				"""
				**Data source:** Yahoo Finance via yfinance.

				**Adjusted close:** Used for all prices to capture splits and dividends consistently.

				**Simple returns:** All analysis uses daily simple returns, not log returns.

				**Annualization:** Mean daily return is multiplied by 252 and daily volatility by $\sqrt{252}$.

				**Risk-free rate:** The annual rate is converted to a daily rate by dividing by 252 when needed.

				**Benchmark:** ^GSPC is included for comparison only and is excluded from optimization.

				**Constraints:** Optimized portfolios use no-short-selling bounds with weights in [0, 1] and sum of weights equal to 1.

				**Efficient frontier:** Generated with constrained optimization across target returns, not Monte Carlo simulation.
				"""
			)
		return

	prices = bundle.prices
	returns = bundle.returns
	selected_tickers = bundle.selected_tickers
	rf = bundle.risk_free_rate
	benchmark = bundle.benchmark
	asset_returns = returns[selected_tickers]
	benchmark_returns = returns[benchmark]
	available_windows = supported_lookback_windows(len(asset_returns))

	if st.session_state.analysis_cache is None:
		with st.spinner("Preparing cached summary statistics, optimization outputs, and the efficient frontier..."):
			st.session_state.analysis_cache = build_analysis_payload(
				prices=prices,
				returns=returns,
				selected_tickers=selected_tickers,
				benchmark=benchmark,
				risk_free_rate=rf,
			)
	analysis = st.session_state.analysis_cache
	stats_table = analysis["stats_table"]
	asset_returns = analysis["asset_returns"]
	benchmark_returns = analysis["benchmark_returns"]
	cov_matrix = analysis["cov_matrix"]
	corr_matrix = analysis["corr_matrix"]
	equal_weights = analysis["equal_weights"]
	equal_metrics = analysis["equal_metrics"]
	equal_series = analysis["equal_series"]
	gmv_weights_series = analysis["gmv_weights_series"]
	gmv_metrics = analysis["gmv_metrics"]
	gmv_series = analysis["gmv_series"]
	gmv_prc = analysis["gmv_prc"]
	gmv_error = analysis["gmv_error"]
	tangency_weights_series = analysis["tangency_weights_series"]
	tangency_metrics = analysis["tangency_metrics"]
	tangency_series = analysis["tangency_series"]
	tangency_prc = analysis["tangency_prc"]
	tan_error = analysis["tan_error"]
	frontier = analysis["frontier"]
	frontier_error = analysis["frontier_error"]
	benchmark_metrics = analysis["benchmark_metrics"]

	with tabs[1]:
		st.subheader("Exploratory Analysis")
		st.write("This section summarizes return behavior, wealth growth, and distribution shape for each security.")

		st.markdown("### Summary Statistics")
		st.dataframe(stats_table.style.format({
			"Annualized Mean Return": "{:.2%}",
			"Annualized Volatility": "{:.2%}",
			"Skewness": "{:.3f}",
			"Kurtosis": "{:.3f}",
			"Minimum Daily Return": "{:.2%}",
			"Maximum Daily Return": "{:.2%}",
			"Sharpe Ratio": "{:.3f}",
			"Sortino Ratio": "{:.3f}",
		}), use_container_width=True)

		st.markdown("### Cumulative Wealth")
		wealth_selection = st.multiselect(
			"Choose series for the $10,000 wealth chart",
			options=selected_tickers + [benchmark],
			default=selected_tickers + [benchmark],
		)
		if wealth_selection:
			wealth_fig = create_cumulative_wealth_chart(prices, wealth_selection)
			st.plotly_chart(wealth_fig, use_container_width=True)
		else:
			st.warning("Select at least one series to display the cumulative wealth chart.")

		st.markdown("### Return Distribution Diagnostics")
		selected_distribution_stock = st.selectbox("Choose a stock for distribution analysis", options=selected_tickers)
		distribution_mode = st.radio(
			"Display mode",
			options=["Histogram with Normal Overlay", "Q-Q Plot"],
			horizontal=True,
		)
		selected_series = asset_returns[selected_distribution_stock].dropna()
		if distribution_mode == "Histogram with Normal Overlay":
			st.plotly_chart(create_distribution_chart(selected_series), use_container_width=True)
		else:
			st.plotly_chart(create_qq_plot(selected_series), use_container_width=True)

	with tabs[2]:
		st.subheader("Risk & Correlation")
		st.write("Inspect rolling volatility, drawdowns, and cross-asset correlations.")

		vol_window = st.select_slider("Rolling volatility window (days)", options=[30, 60, 90, 120], value=60)
		st.plotly_chart(create_rolling_volatility_chart(asset_returns, vol_window), use_container_width=True)

		st.markdown("### Drawdown Analysis")
		drawdown_stock = st.selectbox(
			"Choose a stock for drawdown analysis",
			options=selected_tickers,
			key="drawdown_stock",
		)
		drawdown_fig, max_dd = create_drawdown_chart(asset_returns[drawdown_stock].dropna().rename(drawdown_stock))
		st.metric("Maximum Drawdown", format_percent(max_dd))
		st.plotly_chart(drawdown_fig, use_container_width=True)

		st.markdown("### Risk-Adjusted Metrics")
		risk_table = stats_table[["Sharpe Ratio", "Sortino Ratio"]]
		st.dataframe(risk_table.style.format({"Sharpe Ratio": "{:.3f}", "Sortino Ratio": "{:.3f}"}), use_container_width=True)

		st.markdown("### Correlation Heatmap")
		st.plotly_chart(create_heatmap(corr_matrix, "Annotated Correlation Heatmap of Daily Returns"), use_container_width=True)

		st.markdown("### Rolling Correlation")
		corr_col1, corr_col2, corr_col3 = st.columns(3)
		with corr_col1:
			corr_asset_a = st.selectbox("First stock", options=selected_tickers, key="corr_asset_a")
		with corr_col2:
			available_b = [ticker for ticker in selected_tickers if ticker != corr_asset_a]
			corr_asset_b = st.selectbox("Second stock", options=available_b, key="corr_asset_b")
		with corr_col3:
			corr_window = st.number_input("Rolling window length", min_value=20, max_value=252, value=60, step=5)
		st.plotly_chart(
			create_rolling_correlation_chart(asset_returns[corr_asset_a], asset_returns[corr_asset_b], int(corr_window)),
			use_container_width=True,
		)

		with st.expander("Covariance Matrix"):
			st.dataframe(covariance_matrix(asset_returns).style.format("{:.6f}"), use_container_width=True)

	with tabs[3]:
		st.subheader("Portfolio Optimization")
		st.write("This section compares equal-weight, global minimum variance, tangency, and custom portfolios using only selected stocks.")
		optimization_returns = asset_returns.dropna()
		cov_daily = cov_matrix.to_numpy()

		st.markdown("### Equal-Weight Portfolio")
		eq_cols = st.columns(5)
		for col, label, value in zip(
			eq_cols,
			["Return", "Volatility", "Sharpe", "Sortino", "Max Drawdown"],
			[
				equal_metrics["annual_return"],
				equal_metrics["annual_volatility"],
				equal_metrics["sharpe_ratio"],
				equal_metrics["sortino_ratio"],
				equal_metrics["max_drawdown"],
			],
		):
			with col:
				st.metric(label, format_percent(value) if label in {"Return", "Volatility", "Max Drawdown"} else format_number(value))

		st.plotly_chart(create_weights_bar_chart(equal_weights, "Equal-Weight Portfolio Allocation"), use_container_width=True)

		if gmv_weights_series is not None and gmv_prc is not None:
			st.markdown("### Global Minimum Variance Portfolio")
			gmv_cols = st.columns(5)
			for col, label, value in zip(
				gmv_cols,
				["Return", "Volatility", "Sharpe", "Sortino", "Max Drawdown"],
				[
					gmv_metrics["annual_return"],
					gmv_metrics["annual_volatility"],
					gmv_metrics["sharpe_ratio"],
					gmv_metrics["sortino_ratio"],
					gmv_metrics["max_drawdown"],
				],
			):
				with col:
					st.metric(label, format_percent(value) if label in {"Return", "Volatility", "Max Drawdown"} else format_number(value))
			st.plotly_chart(create_weights_bar_chart(gmv_weights_series, "GMV Portfolio Weights"), use_container_width=True)
			st.plotly_chart(create_risk_contribution_chart(gmv_prc, "GMV Percentage Risk Contributions"), use_container_width=True)
			st.caption(f"GMV PRC sum: {gmv_prc.sum():.4f}")
			st.write("Risk contribution shows how much each asset contributes to total portfolio variance. Higher PRC means that asset explains a larger share of portfolio risk.")

		if tangency_weights_series is not None and tangency_prc is not None:
			st.markdown("### Tangency Portfolio")
			tan_cols = st.columns(5)
			for col, label, value in zip(
				tan_cols,
				["Return", "Volatility", "Sharpe", "Sortino", "Max Drawdown"],
				[
					tangency_metrics["annual_return"],
					tangency_metrics["annual_volatility"],
					tangency_metrics["sharpe_ratio"],
					tangency_metrics["sortino_ratio"],
					tangency_metrics["max_drawdown"],
				],
			):
				with col:
					st.metric(label, format_percent(value) if label in {"Return", "Volatility", "Max Drawdown"} else format_number(value))
			st.plotly_chart(create_weights_bar_chart(tangency_weights_series, "Tangency Portfolio Weights"), use_container_width=True)
			st.plotly_chart(create_risk_contribution_chart(tangency_prc, "Tangency Portfolio Percentage Risk Contributions"), use_container_width=True)
			st.caption(f"Tangency PRC sum: {tangency_prc.sum():.4f}")
			st.write("Risk contribution is computed as $PRC_i = w_i (\Sigma w)_i / (w^T \Sigma w)$ using the full quadratic form of portfolio variance.")

		st.markdown("### Custom Portfolio")
		st.write("Set an independent slider for each stock, then the app normalizes the weights before calculating metrics.")
		slider_columns = st.columns(2 if len(selected_tickers) <= 6 else 3)
		raw_slider_weights: dict[str, float] = {}
		for idx, ticker in enumerate(selected_tickers):
			with slider_columns[idx % len(slider_columns)]:
				raw_slider_weights[ticker] = st.slider(
					f"{ticker} weight setting",
					min_value=0.0,
					max_value=CUSTOM_WEIGHT_MAX,
					value=float(CUSTOM_WEIGHT_MAX / len(selected_tickers)),
					step=1.0,
					key=f"custom_slider_{ticker}",
				)

		custom_weights, custom_total = normalize_slider_weights(raw_slider_weights)
		if custom_total <= 0:
			st.warning("All custom sliders are zero, so the app is reverting to equal weights for the custom portfolio.")
		st.caption(f"Raw slider total before normalization: {custom_total:.1f}")
		st.dataframe(
			pd.DataFrame({"Normalized Weight": custom_weights, "Raw Slider Value": pd.Series(raw_slider_weights)}).style.format(
				{"Normalized Weight": "{:.2%}", "Raw Slider Value": "{:.1f}"}
			),
			use_container_width=True,
		)

		custom_series = portfolio_daily_series(custom_weights.to_numpy(), optimization_returns)
		custom_metrics = portfolio_metrics(custom_weights.to_numpy(), optimization_returns, rf)
		custom_cols = st.columns(5)
		for col, label, value in zip(
			custom_cols,
			["Return", "Volatility", "Sharpe", "Sortino", "Max Drawdown"],
			[
				custom_metrics["annual_return"],
				custom_metrics["annual_volatility"],
				custom_metrics["sharpe_ratio"],
				custom_metrics["sortino_ratio"],
				custom_metrics["max_drawdown"],
			],
		):
			with col:
				st.metric(label, format_percent(value) if label in {"Return", "Volatility", "Max Drawdown"} else format_number(value))

		st.plotly_chart(create_weights_bar_chart(custom_weights, "Custom Portfolio Weights"), use_container_width=True)

		if frontier_error:
			st.warning(frontier_error)

		portfolio_points = {
			"equal": {"return": equal_metrics["annual_return"], "volatility": equal_metrics["annual_volatility"]},
			"benchmark": {"return": benchmark_metrics["annual_return"], "volatility": benchmark_metrics["annual_volatility"]},
			"custom": {"return": custom_metrics["annual_return"], "volatility": custom_metrics["annual_volatility"]},
		}
		if gmv_metrics is not None:
			portfolio_points["gmv"] = {"return": gmv_metrics["annual_return"], "volatility": gmv_metrics["annual_volatility"]}
		if tangency_metrics is not None:
			portfolio_points["tangency"] = {
				"return": tangency_metrics["annual_return"],
				"volatility": tangency_metrics["annual_volatility"],
				"sharpe_ratio": tangency_metrics["sharpe_ratio"],
			}

		frontier_chart = create_frontier_chart(
			frontier=frontier,
			asset_metrics=stats_table.loc[selected_tickers, ["Annualized Mean Return", "Annualized Volatility"]],
			benchmark_point=portfolio_points["benchmark"],
			equal_point=portfolio_points["equal"],
			gmv_point=portfolio_points.get("gmv"),
			tangency_point=portfolio_points.get("tangency"),
			custom_point=portfolio_points["custom"],
			annual_risk_free_rate=rf,
		)
		st.plotly_chart(frontier_chart, use_container_width=True)
		st.info(
			"The efficient frontier shows the minimum achievable volatility for each target return under no-short-selling constraints. The Capital Allocation Line connects the risk-free rate to the tangency portfolio."
		)

		st.markdown("### Portfolio Comparison")
		comparison_metrics = {
			"Equal Weight": equal_metrics,
			"Benchmark": benchmark_metrics,
			"Custom": custom_metrics,
		}
		comparison_weights = {
			"Equal Weight": equal_weights,
			"Benchmark": pd.Series({ticker: np.nan for ticker in selected_tickers}),
			"Custom": custom_weights,
		}
		if gmv_metrics is not None and gmv_weights_series is not None:
			comparison_metrics["GMV"] = gmv_metrics
			comparison_weights["GMV"] = gmv_weights_series
		if tangency_metrics is not None and tangency_weights_series is not None:
			comparison_metrics["Tangency"] = tangency_metrics
			comparison_weights["Tangency"] = tangency_weights_series

		portfolio_table = build_portfolio_table(comparison_metrics, comparison_weights)
		st.dataframe(
			portfolio_table.style.format(
				{
					"Annualized Return": "{:.2%}",
					"Annualized Volatility": "{:.2%}",
					"Sharpe Ratio": "{:.3f}",
					"Sortino Ratio": "{:.3f}",
					"Maximum Drawdown": "{:.2%}",
				}
			),
			use_container_width=True,
		)

		comparison_series: dict[str, pd.Series] = {
			"Equal Weight": equal_series,
			"Custom": custom_series,
			BENCHMARK_TICKER: benchmark_returns,
		}
		if gmv_series is not None:
			comparison_series["GMV"] = gmv_series
		if tangency_series is not None:
			comparison_series["Tangency"] = tangency_series

		comparison_df = pd.DataFrame(comparison_series).dropna(how="any")
		if not comparison_df.empty:
			wealth_chart = go.Figure()
			for column in comparison_df.columns:
				wealth_chart.add_trace(go.Scatter(x=comparison_df.index, y=wealth_index(comparison_df[column]), mode="lines", name=column))
			wealth_chart.update_layout(
				title="Cumulative Wealth Comparison of Portfolios and Benchmark",
				xaxis_title="Date",
				yaxis_title="Wealth Value ($)",
				height=600,
			)
			st.plotly_chart(wealth_chart, use_container_width=True)

	with tabs[4]:
		st.subheader("Estimation Window Sensitivity")
		st.write("Optimization results can change materially when the estimation window changes, so this section compares multiple lookback horizons.")

		selected_window_labels = st.multiselect(
			"Lookback windows",
			options=list(available_windows.keys()),
			default=list(available_windows.keys()),
		)

		with st.spinner("Evaluating estimation window sensitivity..."):
			sensitivity_rows = []
			gmv_weight_columns: dict[str, pd.Series] = {}
			tan_weight_columns: dict[str, pd.Series] = {}
			for label in selected_window_labels:
				days = available_windows[label]
				window_payload = compute_window_portfolios(optimization_returns, rf, days)
				window_returns = window_payload["window_returns"]
				gmv_w = window_payload["gmv_weights"]
				tan_w = window_payload["tangency_weights"]
				gmv_err = window_payload["gmv_error"]
				tan_err = window_payload["tan_error"]
				if len(window_returns) < MIN_TICKERS + 10:
					st.warning(f"{label} does not have enough observations after cleaning and was skipped.")
					continue
				if gmv_w is None or tan_w is None:
					st.warning(f"{label}: {gmv_err or tan_err}")
					continue

				gmv_metrics_window = portfolio_metrics(gmv_w, window_returns, rf)
				tan_metrics_window = portfolio_metrics(tan_w, window_returns, rf)

				sensitivity_rows.append(
					{
						"Window": label,
						"GMV Annualized Return": gmv_metrics_window["annual_return"],
						"GMV Annualized Volatility": gmv_metrics_window["annual_volatility"],
						"Tangency Annualized Return": tan_metrics_window["annual_return"],
						"Tangency Annualized Volatility": tan_metrics_window["annual_volatility"],
						"Tangency Sharpe": tan_metrics_window["sharpe_ratio"],
					}
				)
				gmv_weight_columns[label] = pd.Series(gmv_w, index=selected_tickers)
				tan_weight_columns[label] = pd.Series(tan_w, index=selected_tickers)

		if sensitivity_rows:
			sensitivity_table = pd.DataFrame(sensitivity_rows).set_index("Window")
			st.dataframe(
				sensitivity_table.style.format(
					{
						"GMV Annualized Return": "{:.2%}",
						"GMV Annualized Volatility": "{:.2%}",
						"Tangency Annualized Return": "{:.2%}",
						"Tangency Annualized Volatility": "{:.2%}",
						"Tangency Sharpe": "{:.3f}",
					}
				),
				use_container_width=True,
			)

			st.markdown("### Grouped Weight Comparison")
			if gmv_weight_columns:
				gmv_weight_df = pd.DataFrame(gmv_weight_columns)
				gmv_fig = go.Figure()
				for asset in gmv_weight_df.index:
					gmv_fig.add_trace(go.Bar(name=asset, x=gmv_weight_df.columns, y=gmv_weight_df.loc[asset].values))
				gmv_fig.update_layout(
					title="GMV Weights Across Estimation Windows",
					xaxis_title="Estimation Window",
					yaxis_title="Weight",
					barmode="group",
					height=550,
				)
				st.plotly_chart(gmv_fig, use_container_width=True)

			if tan_weight_columns:
				tan_weight_df = pd.DataFrame(tan_weight_columns)
				tan_fig = go.Figure()
				for asset in tan_weight_df.index:
					tan_fig.add_trace(go.Bar(name=asset, x=tan_weight_df.columns, y=tan_weight_df.loc[asset].values))
				tan_fig.update_layout(
					title="Tangency Weights Across Estimation Windows",
					xaxis_title="Estimation Window",
					yaxis_title="Weight",
					barmode="group",
					height=550,
				)
				st.plotly_chart(tan_fig, use_container_width=True)

			st.info(
				"Optimization is sensitive to the estimation window because mean returns and covariances are noisy estimates. Changing the sample period can materially shift both the frontier and the weights."
			)
		else:
			st.warning("No supported estimation windows were available for the current data range.")

	with tabs[5]:
		st.subheader("About / Methodology")
		st.markdown(
			"""
			**Data source**  
			Prices are downloaded from Yahoo Finance using yfinance.

			**Adjusted close usage**  
			All calculations use adjusted close prices only so splits and dividends are incorporated consistently.

			**Simple return convention**  
			Every analysis uses daily simple returns computed with percentage change.

			**Annualization assumptions**  
			Mean daily return is annualized by multiplying by 252. Daily volatility is annualized by multiplying by $\sqrt{252}$.

			**Risk-free rate handling**  
			The user enters an annualized risk-free rate. When a daily target is needed for Sortino calculations, it is divided by 252.

			**Benchmark treatment**  
			The S&P 500 benchmark (^GSPC) is included for comparison but excluded from optimization.

			**No-short-selling constraints**  
			Optimized portfolios are solved with weights constrained to the interval [0, 1] and a full-weight sum constraint of 1.

			**Efficient frontier method**  
			The efficient frontier is built with constrained optimization across target returns. It is not estimated through random portfolio simulation.
			"""
		)


if __name__ == "__main__":
	main()
