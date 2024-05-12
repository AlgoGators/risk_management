import pandas as pd
from scipy.stats import norm

class _utils:
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

class RiskMeasures:
    def __init__(
            self,
            trend_tables : dict[str, pd.DataFrame],
            weights : tuple[float, float, float] = (0.01, 0.01, 0.98),
            warmup : int = 100,
            unadj_column : str = "Unadj_Close",
            expiration_column : str = "Delivery Month",
            date_column : str = "Date",
            fill : bool = True) -> None:

        self.weights = weights
        self.warmup = warmup
        self.trend_tables : dict[str, pd.DataFrame] = trend_tables
        self.unadj_column : str = unadj_column
        self.expiration_column : str = expiration_column
        self.date_column : str = date_column
        self.fill : bool = fill

        self.daily_returns : pd.DataFrame
        self.product_returns : pd.DataFrame
        self.GARCH_variances : pd.DataFrame
        self.GARCH_covariances : pd.DataFrame

    def construct(self) -> None:
        self.daily_returns = self.__calculate_daily_returns()
        self.product_returns = self.__calculate_product_returns()
        self.GARCH_variances = self.__calculate_GARCH_variances()
        self.GARCH_covariances = self.__calculate_GARCH_covariances()

        self.reindex(True)

    def reindex(self, inner : bool = True) -> None:
        if inner:
            indexes = self.daily_returns.index.intersection(self.product_returns.index).intersection(self.GARCH_variances.index).intersection(self.GARCH_covariances.index)
            self.daily_returns = self.daily_returns.loc[indexes]
            self.product_returns = self.product_returns.loc[indexes]
            self.GARCH_variances = self.GARCH_variances.loc[indexes]
            self.GARCH_covariances = self.GARCH_covariances.loc[indexes]
            return
        
        indexes = self.daily_returns.index.union(self.product_returns.index).union(self.GARCH_variances.index).union(self.GARCH_covariances.index)
        self.daily_returns = self.daily_returns.reindex(indexes)
        self.product_returns = self.product_returns.reindex(indexes)
        self.GARCH_variances = self.GARCH_variances.reindex(indexes)
        self.GARCH_covariances = self.GARCH_covariances.reindex(indexes)


    # ? def update(self, product_returns, GARCH_variances, GARCH_covariances) -> None:
    # ?     self.daily_returns = self.__calculate_daily_returns(self.trend_tables)
    # ?     self.__update_product_returns(product_returns, self.daily_returns)
    # ?     self.__update_GARCH_variance(GARCH_variances, self.daily_returns)
    # ?     self.__update_GARCH_covariance(GARCH_covariances, product_returns)

    def __calculate_daily_returns(self) -> dict[str, pd.DataFrame]:
        daily_returns : pd.DataFrame = pd.DataFrame()
        for i, instrument in enumerate(list(self.trend_tables.keys())):
            prices = self.trend_tables[instrument]
            # creates a set of unique delivery months
            delivery_months: set = set(prices[self.expiration_column].tolist())
            # converts back to list for iterating
            delivery_months: list = list(delivery_months)
            delivery_months.sort()

            percent_returns = pd.DataFrame()

            for i, delivery_month in enumerate(delivery_months):
                    
                # creates a dataframe for each delivery month
                df_delivery_month = prices[prices[self.expiration_column] == delivery_month]

                percent_change = pd.DataFrame()
                percent_change[instrument] = df_delivery_month[self.unadj_column].diff() / df_delivery_month[self.unadj_column].abs().shift()

                # set index
                if self.date_column in df_delivery_month.columns:
                    dates = df_delivery_month[self.date_column]
                else:
                    dates = df_delivery_month.index
                    
                percent_change.index = dates
                percent_change.index.name = self.date_column

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

        daily_returns = _utils.ffill_zero(daily_returns) if self.fill else daily_returns

        return daily_returns
    
    def __calculate_product_returns(self) -> pd.DataFrame:
        instruments = self.daily_returns.columns.tolist()
        instruments.sort()

        product_returns = pd.DataFrame()

        product_dictionary : dict = {}

        for i, instrument_X in enumerate(instruments):
            for j, instrument_Y in enumerate(instruments):
                if i > j:
                    continue
                
                product_dictionary[f'{instrument_X}_{instrument_Y}'] = self.daily_returns[instrument_X] * self.daily_returns[instrument_Y]

        product_returns = pd.DataFrame(product_dictionary, index=self.daily_returns.index)

        product_returns = _utils.ffill_zero(product_returns) if self.fill else product_returns

        return product_returns

    #? def __update_product_returns(product_returns : pd.DataFrame, returns : pd.DataFrame) -> pd.DataFrame:
    #?     instruments = returns.columns.tolist()
    #?     instruments.sort()

    #?     product_dictionary : dict = {}

    #?     for i, instrument_X in enumerate(instruments):
    #?         for j, instrument_Y in enumerate(instruments):
    #?             if i > j:
    #?                 continue
     
    #?             product_dictionary[f'{instrument_X}_{instrument_Y}'] = returns[instrument_X].iloc[-1] * returns[instrument_Y].iloc[-1]

    #?     product_returns.loc[returns.index[-1]] = pd.Series(product_dictionary)

    #?     return product_returns

    def __calculate_GARCH_variance(self, squared_return : float, last_estimate : float, LT_variance : float) -> float:
        if sum(self.weights) != 1:
            raise ValueError('The sum of the weights must be equal to 1')

        return squared_return * self.weights[0] + last_estimate * self.weights[1] + LT_variance * self.weights[2]

    def __calculate_GARCH_variances(self) -> pd.DataFrame:
        if sum(self.weights) != 1:
            raise ValueError('The sum of the weights must be equal to 1')
        
        GARCH_variances : pd.DataFrame = pd.DataFrame()

        for i, instrument in enumerate(self.daily_returns.columns.tolist()):
            squared_returns = self.daily_returns[instrument] ** 2
            squared_returns.dropna(inplace=True)

            dates = squared_returns.index

            # Calculate rolling LT variance
            LT_variances = squared_returns.rolling(window=self.warmup).mean().fillna(method='bfill')

            df = pd.Series(index=dates)
            df[0] = squared_returns[0]

            for j, _ in enumerate(dates[1:], 1):
                df[j] = self.__calculate_GARCH_variance(squared_returns[j], df[j-1], LT_variances[j])

            if i == 0:
                GARCH_variances = df.to_frame(instrument)
                continue

            GARCH_variances = pd.merge(GARCH_variances, df.to_frame(instrument), how='outer', left_index=True, right_index=True)

        GARCH_variances = GARCH_variances.interpolate() if self.fill else GARCH_variances

        return GARCH_variances[self.warmup:]

    #? def __update_GARCH_variance(self, variances : pd.DataFrame, returns : pd.DataFrame) -> pd.DataFrame:
    #?     returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')
    #?     GARCH_variances = {}

    #?     # Calculate the GARCH variances
    #?     for instrument in returns.columns.tolist():
    #?         squared_returns = returns[instrument].iloc[-self.warmup:] ** 2
    #?         squared_return = squared_returns.iloc[-1]
    #?         last_estimate = variances[instrument].iloc[-1]
    #?         LT_variance = squared_returns.mean()
    #?         GARCH_variances[instrument] = self.__calculate_GARCH_variance(squared_return, last_estimate, LT_variance, self.weights)

    #?     # Could be removed if we want the row instead to append to the database
    #?     variances.loc[returns.index[-1]] = pd.Series(GARCH_variances)
    #?     return variances

    def __calculate_GARCH_covariances(self) -> pd.DataFrame:
        if sum(self.weights) != 1:
            raise ValueError('The sum of the weights must be equal to 1')
        
        # GARCH_covariances : pd.DataFrame = pd.DataFrame()

        product_returns = self.product_returns.dropna()
        LT_covariances : pd.DataFrame = product_returns.rolling(window=self.warmup).mean().fillna(method='bfill')

        LT_covars = LT_covariances.values
        p_returns = product_returns.values

        GARCH_covariances = pd.DataFrame(index=product_returns.index, columns=product_returns.columns, dtype=float)
        GARCH_covariances.iloc[0] = p_returns[0]

        for i in range(1, len(p_returns)):
            GARCH_covariances.iloc[i] = p_returns[i] * self.weights[0] + GARCH_covariances.iloc[i-1] * self.weights[1] + LT_covars[i] * self.weights[2]

        GARCH_covariances = GARCH_covariances.interpolate() if self.fill else GARCH_covariances

        return GARCH_covariances.iloc[self.warmup:, :]

    #? def __update_GARCH_covariance(self, covariances : pd.DataFrame, product_returns : pd.DataFrame) -> pd.DataFrame:
    #?     if sum(self.weights) != 1:
    #?         raise ValueError('The sum of the weights must be equal to 1')

    #?     covariances.loc[product_returns.index[-1]] = product_returns.iloc[-1] * self.weights[0] + covariances.iloc[-1] * self.weights[1] + product_returns.iloc[-self.warmup:].mean() * self.weights[2]
    #?     return covariances


def calculate_value_at_risk_historical(returns : pd.DataFrame, confidence_level : float, lookback : int) -> pd.DataFrame:
    # Calculate the historical value at risk
    return -returns.rolling(window=lookback).quantile(1 - confidence_level)

def calculate_value_at_risk_parametric(variances : pd.DataFrame, confidence_level : float) -> pd.DataFrame:
    # Calculate the parametric value at risk
    return -variances.apply(lambda x: x ** 0.5 * norm.ppf(1 - confidence_level))
