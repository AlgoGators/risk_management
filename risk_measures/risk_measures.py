import pandas as pd
import numpy as np
from scipy.stats import norm

def ffill_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill zeros in a DataFrame. This function will replace all zeros with the last non-zero value in the DataFrame.
    """
    # We assume any gaps in percent returns at this point are because the market was closed that day,
    # but only fill forward; 
    #* apologies for the complexity but it was the only way i could find
    # Find the index of the first non-NaN value in each column
    first_non_nan_index = df.apply(lambda x: x.first_valid_index())

    # Iterate over each column and replace NaN values below the first non-NaN index with 0

    # Iterate over each column and replace NaN values below the first non-NaN index with 0
    for column in df.columns:
        first_index = first_non_nan_index[column]
        df.loc[first_index:, column] = df.loc[first_index:, column].fillna(0)

    return df

def calculate_daily_returns(
        trend_tables : dict[str, pd.DataFrame], 
        unadj_column : str, 
        expiration_column : str, 
        date_column : str, 
        fill : bool = False) -> dict[str, pd.DataFrame]:

    daily_returns : pd.DataFrame = pd.DataFrame()
    for i, instrument in enumerate(list(trend_tables.keys())):
        prices = trend_tables[instrument]
        # creates a set of unique delivery months
        delivery_months: set = set(prices[expiration_column].tolist())
        # converts back to list for iterating
        delivery_months: list = list(delivery_months)
        delivery_months.sort()

        percent_returns = pd.DataFrame()

        for i, delivery_month in enumerate(delivery_months):
                
            # creates a dataframe for each delivery month
            df_delivery_month = prices[prices[expiration_column] == delivery_month]

            percent_change = pd.DataFrame()
            percent_change[instrument] = df_delivery_month[unadj_column].diff() / df_delivery_month[unadj_column].abs().shift()

            # set index
            if date_column in df_delivery_month.columns:
                dates = df_delivery_month[date_column]
            else:
                dates = df_delivery_month.index
                
            percent_change.index = dates
            percent_change.index.name = date_column

            delivery_month_returns = percent_change

            if i != 0:
                # replace the NaN with 0
                delivery_month_returns.fillna(0, inplace=True)

            #? Might be worth duplicating the last % return to include the missing day for the roll
            percent_returns = pd.concat([percent_returns, delivery_month_returns])

        if i == 0:
            daily_returns = percent_returns
            continue

        daily_returns = pd.merge(daily_returns, percent_returns, how='outer', left_index=True, right_index=True)

    daily_returns = ffill_zero(daily_returns) if fill else daily_returns

    return daily_returns

def calculate_weekly_returns(daily_returns : pd.DataFrame, fill : bool = False) -> pd.DataFrame:
    # add 1 to each return to get the multiplier
    return_multipliers = daily_returns + 1

    # ! there is some statistical error here because the first weekly return could be
    # ! less than 5 returns but this should be insignificant for the quantity of returns
    n = len(return_multipliers)
    complete_groups_index = n // 5 * 5  # This will give the largest number less than n that is a multiple of 5
    sliced_return_multipliers = return_multipliers[:complete_groups_index]

    # group into chunks of 5, and then calculate the product of each chunk, - 1 to get the return
    weekly_returns = sliced_return_multipliers.groupby(np.arange(complete_groups_index) // 5).prod() - 1

    # Use the last date in each chunk
    weekly_returns.index = daily_returns.index[4::5]

    nan_mask = daily_returns.isna().copy()

    weekly_returns = weekly_returns.where(~nan_mask.iloc[4::5])

    weekly_returns = ffill_zero(weekly_returns) if fill else weekly_returns

    return weekly_returns

def calculate_product_returns(returns : pd.DataFrame, fill=False) -> pd.DataFrame:
    instruments = returns.columns.tolist()
    instruments.sort()

    product_returns = pd.DataFrame()

    product_dictionary : dict = {}

    for i, instrument_X in enumerate(instruments):
        for j, instrument_Y in enumerate(instruments):
            if i > j:
                continue
            
            product_dictionary[f'{instrument_X}_{instrument_Y}'] = returns[instrument_X] * returns[instrument_Y]

    product_returns = pd.DataFrame(product_dictionary, index=returns.index)

    product_returns = ffill_zero(product_returns) if fill else product_returns

    return product_returns

def update_product_returns(product_returns : pd.DataFrame, returns : pd.DataFrame) -> pd.DataFrame:
    instruments = returns.columns.tolist()
    instruments.sort()

    product_dictionary : dict = {}

    for i, instrument_X in enumerate(instruments):
        for j, instrument_Y in enumerate(instruments):
            if i > j:
                continue
            
            product_dictionary[f'{instrument_X}_{instrument_Y}'] = returns[instrument_X].iloc[-1] * returns[instrument_Y].iloc[-1]

    product_returns.loc[returns.index[-1]] = pd.Series(product_dictionary)

    return product_returns

def calculate_GARCH_variance(squared_return : float, last_estimate : float, LT_variance : float, weights : tuple[float, float, float]) -> float:
    if sum(weights) != 1:
        raise ValueError('The sum of the weights must be equal to 1')

    return squared_return * weights[0] + last_estimate * weights[1] + LT_variance * weights[2]

def calculate_GARCH_variances(returns : pd.DataFrame, warmup : int, weights : tuple[float, float, float], fill=False) -> pd.DataFrame:
    if sum(weights) != 1:
        raise ValueError('The sum of the weights must be equal to 1')
    
    GARCH_variances : pd.DataFrame = pd.DataFrame()

    for i, instrument in enumerate(returns.columns.tolist()):
        squared_returns = returns[instrument] ** 2
        squared_returns.dropna(inplace=True)

        dates = squared_returns.index

        # Calculate rolling LT variance
        LT_variances = squared_returns.rolling(window=warmup).mean().fillna(method='bfill')

        df = pd.Series(index=dates)
        df[0] = squared_returns[0]

        for j, _ in enumerate(dates[1:], 1):
            df[j] = calculate_GARCH_variance(squared_returns[j], df[j-1], LT_variances[j], weights)

        if i == 0:
            GARCH_variances = df.to_frame(instrument)
            continue

        GARCH_variances = pd.merge(GARCH_variances, df.to_frame(instrument), how='outer', left_index=True, right_index=True)

    GARCH_variances = GARCH_variances.interpolate() if fill else GARCH_variances

    return GARCH_variances[warmup:]

def update_GARCH_variance(variances : pd.DataFrame, returns : pd.DataFrame, warmup : int, weights : tuple[float, float, float]) -> pd.DataFrame:
    returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')
    GARCH_variances = {}

    # Calculate the GARCH variances
    for instrument in returns.columns.tolist():
        squared_returns = returns[instrument].iloc[-warmup:] ** 2
        squared_return = squared_returns.iloc[-1]
        last_estimate = variances[instrument].iloc[-1]
        LT_variance = squared_returns.mean()
        GARCH_variances[instrument] = calculate_GARCH_variance(squared_return, last_estimate, LT_variance, weights)

    # Could be removed if we want the row instead to append to the database
    variances.loc[returns.index[-1]] = pd.Series(GARCH_variances)
    return variances

def calculate_GARCH_covariance(pair_return : float, last_estimate : float, LT_covariance : float, weights : tuple[float, float, float]) -> float:
    if sum(weights) != 1:
        raise ValueError('The sum of the weights must be equal to 1')

    return pair_return * weights[0] + last_estimate * weights[1] + LT_covariance * weights[2]

def calculate_GARCH_covariances(product_returns : pd.DataFrame, warmup : int, weights : tuple[float, float, float], fill=False) -> pd.DataFrame:
    if sum(weights) != 1:
        raise ValueError('The sum of the weights must be equal to 1')
    
    # GARCH_covariances : pd.DataFrame = pd.DataFrame()

    product_returns.dropna(inplace=True)
    LT_covarariances : pd.DataFrame = product_returns.rolling(window=warmup).mean().fillna(method='bfill')

    LT_covars = LT_covarariances.values
    p_returns = product_returns.values

    GARCH_covariances = pd.DataFrame(index=product_returns.index, columns=product_returns.columns, dtype=float)
    GARCH_covariances.iloc[0] = p_returns[0]

    for i in range(1, len(p_returns)):
        GARCH_covariances.iloc[i] = p_returns[i] * weights[0] + GARCH_covariances.iloc[i-1] * weights[1] + LT_covars[i] * weights[2]

    GARCH_covariances = GARCH_covariances.interpolate() if fill else GARCH_covariances

    return GARCH_covariances.iloc[warmup:, :]

def update_GARCH_covariance(covariances : pd.DataFrame, product_returns : pd.DataFrame, warmup : int, weights : tuple[float, float, float]) -> pd.DataFrame:
    GARCH_covariances = {}

    # Calculate the GARCH covariances
    for pair in product_returns.columns.tolist():
        pair_returns = product_returns[pair].iloc[-warmup:]
        pair_return = pair_returns.iloc[-1]
        last_estimate = covariances[pair].iloc[-1]
        LT_covariance = pair_returns.mean()
        GARCH_covariances[pair] = calculate_GARCH_covariance(pair_return, last_estimate, LT_covariance, weights)

    # Could be removed if we want the row instead to append to the database
    covariances.loc[product_returns.index[-1]] = pd.Series(GARCH_covariances)
    return covariances

def calculate_value_at_risk_historical(returns : pd.DataFrame, confidence_level : float, lookback : int) -> pd.DataFrame:
    # Calculate the historical value at risk
    return -returns.rolling(window=lookback).quantile(1 - confidence_level)

def calculate_value_at_risk_parametric(variances : pd.DataFrame, confidence_level : float) -> pd.DataFrame:
    # Calculate the parametric value at risk
    return -variances.apply(lambda x: x ** 0.5 * norm.ppf(1 - confidence_level))
