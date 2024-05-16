import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import yfinance

def calculate_PNL(positions : pd.Series, prices : pd.Series) -> pd.Series:
    pos_series = positions.groupby(positions.index).last()
    both_series = pd.concat([pos_series, prices], axis=1)
    both_series.columns = ["positions", "prices"]
    both_series = both_series.ffill()

    price_returns = both_series.prices.diff()

    returns = both_series.positions.shift(1) * price_returns

    returns[returns.isna()] = 0.0

    return returns

def calculate_PNL_df(positions : pd.DataFrame, prices : pd.DataFrame, multipliers : pd.DataFrame) -> pd.DataFrame:
    pnl = pd.DataFrame()
    for instrument in positions.columns:
        pnl[instrument] = calculate_PNL(positions[instrument], prices[instrument]) * multipliers[instrument].iloc[0]
    return pnl

def calculate_portfolio_PNL(pnl_df : pd.DataFrame) -> pd.Series:
    return pnl_df.sum(axis=1)

def calculate_PCT_PNL(portfolio, benchmark, prices, multipliers, capital) -> tuple[pd.Series, pd.Series]:
    portfolio_PNLs_df = calculate_PNL_df(portfolio, prices, multipliers)
    benchmark_PNLs_df = calculate_PNL_df(benchmark, prices, multipliers)
    portfolio_returns = calculate_portfolio_PNL(portfolio_PNLs_df)
    benchmark_returns = calculate_portfolio_PNL(benchmark_PNLs_df)
    portfolio_returns = portfolio_returns / capital
    benchmark_returns = benchmark_returns / capital
    return portfolio_returns, benchmark_returns

def equity_curve(portfolio_returns : pd.Series) -> pd.Series:
    return portfolio_returns.cumsum().ffill()

def benchmark_tracking_error(portfolio_returns_PCT : pd.Series, benchmark_returns_PCT : pd.Series) -> float:
    """
    Returns the tracking error between the portfolio and benchmark positions

    Parameters:
    ---
        portfolio : pd.DataFrame
            the portfolio positions
        benchmark : pd.DataFrame
            the benchmark positions
    """
    portfolio_returns_PCT, benchmark_returns_PCT = calculate_PCT_PNL(portfolio, benchmark, prices, multipliers, capital)

    return (((portfolio_returns_PCT - benchmark_returns_PCT) ** 2).sum() / len(portfolio_returns_PCT)) ** 0.5

def plot_equity_curves(portfolio_equity : pd.Series, benchmark_equity : pd.Series, sp500_equity):
    try:
        limits = pd.read_csv('log.csv', index_col=1)
        limits.index = pd.to_datetime(limits.index)
        limits.drop(columns=['Level', 'type', 'additional_info'], inplace=True)

        limits['message'] = limits['subtype'] + ' ' + limits['info']
        limits.drop(columns=['subtype', 'info'], inplace=True)
        plt.scatter(limits.index, portfolio_equity.loc[limits.index], color='black', label='Limits')
    except pd.errors.EmptyDataError:
        pass

    plt.plot(sp500_equity, label='SP500')
    plt.plot(portfolio_equity, label='Portfolio')
    plt.plot(benchmark_equity, label='Benchmark')
    
    # Format x-axis to show dates every few years
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    
    # Set y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}%'))  # 0 decimal places

    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()

def plot_PNL(portfolio_returns : pd.Series, benchmark_returns : pd.Series):
    plt.plot(portfolio_returns, label='Portfolio')
    plt.plot(benchmark_returns, label='Benchmark')
    plt.xlabel('Time')
    plt.ylabel('PNL')
    plt.legend()
    plt.show()

def sharpe_ratio(portfolio):
    return portfolio.mean() / portfolio.std()

def information_ratio(portfolio, benchmark):
    return (portfolio - benchmark).mean() / (portfolio - benchmark).std()

def get_SP500(prices : pd.DataFrame):
    x = yfinance.download(tickers = '^GSPC', start=prices.index[0].strftime('%Y-%m-%d'), end=prices.index[-1].strftime('%Y-%m-%d'))
    x.to_csv('sp500.csv')

if __name__ == "__main__":
    prices = pd.read_parquet('dyn_opt/unittesting/data/adj_prices.parquet')
    benchmark = pd.read_parquet('dyn_opt/unittesting/data/ideal_positions.parquet')
    portfolio = pd.read_parquet('dyn_opt/unittesting/data/optimized_positions.parquet')
    multipliers = pd.read_parquet('dyn_opt/unittesting/data/multipliers.parquet')

    prices.index = pd.to_datetime(prices.index)
    benchmark.index = pd.to_datetime(benchmark.index)
    portfolio.index = pd.to_datetime(portfolio.index)

    prices = prices.reindex(portfolio.index)
    benchmark = benchmark.reindex(portfolio.index)
    capital = 500_000

    portfolio_returns_PCT, benchmark_returns_PCT = calculate_PCT_PNL(portfolio, benchmark, prices, multipliers, capital)
    
    sp500 = pd.read_csv('sp500.csv')
    sp500.index = pd.to_datetime(sp500['Date'])
    sp500 = sp500['Adj Close']
    sp500 = sp500.pct_change().dropna()
    
    sp500_equity = equity_curve(sp500 * 100)
    portfolio_equity = equity_curve(portfolio_returns_PCT *100)
    benchmark_equity = equity_curve(benchmark_returns_PCT *100)

    # plot_PNL(portfolio_returns_PCT, benchmark_returns_PCT)
    plot_equity_curves(portfolio_equity, benchmark_equity, sp500_equity)

    X = sm.add_constant(benchmark_returns_PCT)

    model = sm.OLS(portfolio_returns_PCT, X).fit()

    b0, b1 = model.params

    regression_line = f"y = {b1:.2f}x + {b0:.4f}"
    print(regression_line)

    print(sharpe_ratio(portfolio_returns_PCT))
    print(sharpe_ratio(benchmark_returns_PCT))
    print(information_ratio(portfolio_returns_PCT, benchmark_returns_PCT))

    tracking_error = benchmark_tracking_error(portfolio_returns_PCT, benchmark_returns_PCT)
    print(tracking_error)