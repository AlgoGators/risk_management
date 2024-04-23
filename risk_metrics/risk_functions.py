import pandas as pd
import numpy as np
from typing import Tuple
from math import isclose

class PercentReturns:
    def __init__(
        self,
        prices: pd.DataFrame,
        return_column: str,
        unadj_column: str,
        expiration_column: str,
        date_column: str) -> None:

        self.prices : pd.DataFrame = prices
        self.return_column : str = return_column

        self.unadj_column : str = unadj_column
        self.expiration_column : str = expiration_column
        self.date_column : str = date_column
        

    @property
    def daily_returns(self):
        if not hasattr(self, "_daily_returns"):
            self.__set_instrument_daily_returns()
        return self._daily_returns

    def set_weekly_returns(daily_returns : pd.DataFrame):
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

        return weekly_returns
    
    def __set_instrument_daily_returns(self):
        # creates a set of unique delivery months
        delivery_months: set = set(self.prices[self.expiration_column].tolist())
        # converts back to list for iterating
        delivery_months: list = list(delivery_months)
        delivery_months.sort()

        percent_returns = pd.DataFrame()

        for i, delivery_month in enumerate(delivery_months):
                
            # creates a dataframe for each delivery month
            df_delivery_month = self.prices[self.prices[self.expiration_column] == delivery_month]

            percent_change = pd.DataFrame()
            percent_change[self.return_column] = df_delivery_month[self.unadj_column].diff() / df_delivery_month[self.unadj_column].abs().shift()

            # set index
            percent_change.index = df_delivery_month[self.date_column]
            percent_change.index.name = self.date_column

            delivery_month_returns = percent_change

            if i != 0:
                # replace the NaN with 0
                delivery_month_returns.fillna(0, inplace=True)

            #? Might be worth duplicating the last % return to include the missing day for the roll
            percent_returns = pd.concat([percent_returns, delivery_month_returns])

        self._daily_returns = percent_returns

class Variance:
    def __init__(
            self,
            variances : pd.DataFrame) -> None:
        self.variances : pd.DataFrame = variances

    def __repr__(self) -> str:
        return self.variances.__repr__()

    def df(self) -> pd.DataFrame:
        return self.variances

    def annualize(self):
        return self.variances * 256
    
    def volatility(self):
        return np.sqrt(self.variances)
    
    def to_csv(self, path : str):
        self.variances.to_csv(path)

class LongTermVariance(Variance):
    def LongTermVariance(
            percent_returns : pd.DataFrame,
            percent_return_column : str,
            long_term_variance_column : str,
            date_column : str,
            minimum_observations: int,
            maximum_observations: int) -> float:

        if len(percent_returns) < minimum_observations:
            raise ValueError(f"Length of DataFrame: {len(percent_returns)} is less than {minimum_observations}.")

        if len(percent_returns) < minimum_observations:
            raise ValueError(f"Length of DataFrame: {len(percent_returns)} is less than {minimum_observations}.")

        percent_returns.dropna(inplace=True)

        percent_returns["Square Returns"] = percent_returns[percent_return_column]**2

        # calculate the long-term variance
        first_long_term_variance = percent_returns["Square Returns"][:minimum_observations].mean()

        # this long-term variance is assumed to be the variance for all previous variances
        dates_up_to_minimum_observations = percent_returns.index[:minimum_observations]
        
        # create a DataFrame with all the long-term variances (duplicate the LT variance after minimum observations for all previous observations)
        df : pd.DataFrame = pd.DataFrame({long_term_variance_column: [first_long_term_variance]*(minimum_observations)}, index=dates_up_to_minimum_observations)

        long_term_variances = []
        dates = []

        for i in range(minimum_observations, len(percent_returns)):
            start = i - maximum_observations if i - maximum_observations > 0 else 0
            # use all previous observations to calculate the long-term variance
            long_term_variance = percent_returns["Square Returns"][start:i].mean()

            # appending both value and dates at same time to keep in sync
            long_term_variances.append(long_term_variance)
            dates.append(percent_returns.index[i])

        df = pd.concat([df, pd.DataFrame({long_term_variance_column: long_term_variances}, index=dates)])

        # set index name
        df.index.name = date_column

        return Variance(df)

class GARCHVariance(Variance):
    def __init__(self) -> None:
        self.long_term_variances = None
        self.percent_returns = None
        self.alpha = None
        self.beta = None
        self.gamma = None

    def GARCHVariance(
            self,
            long_term_variances : pd.DataFrame,
            percent_returns : pd.DataFrame,
            weights : Tuple[float, float, float],
            percent_return_column : str,
            long_term_variance_column : str,
            GARCH_variance_column : str,
            date_column : str,
            minimum_observations : int = 256):
        
        self.alpha, self.beta, self.gamma = weights
        if not isclose(self.alpha + self.beta + self.gamma, 1.0, rel_tol=1e-9):
            raise ValueError(f"The sum of the weights: {self.alpha + self.beta + self.gamma} must equal 1.0")
        
        self.long_term_variances = long_term_variances[long_term_variance_column].dropna()
        self.percent_returns = percent_returns[percent_return_column].dropna()


        GARCH_variance = self.GARCHVarianceExplicit(minimum_observations)

        GARCH_variances = []

        for i in range(minimum_observations, len(self.percent_returns)):
            GARCH_variance = self.GARCHVarianceRecursive(GARCH_variance, i)
            GARCH_variances.append(GARCH_variance)

        # Create NaNs for the first minimum_observations
        df = pd.DataFrame({GARCH_variance_column: [np.NaN]*minimum_observations}, index=self.percent_returns.index[0:minimum_observations])

        # Set index name
        df.index.name = date_column

        # Concatenate the GARCH_variances to the DataFrame
        df = pd.concat([df, pd.DataFrame({GARCH_variance_column: GARCH_variances}, index=self.percent_returns.index[minimum_observations:])])

        return Variance(df)
        
    def get_long_term_variances(self, index, k):
        return self.long_term_variances[index-k]
    
    def get_percent_returns(self, index, k):
        return self.percent_returns[index-k]
        
    def GARCHVarianceExplicit(self, index : int = 256):
        variance = 0
        for k in range(0, index-1):
            variance += self.beta**(k) * (self.gamma * self.get_long_term_variances(index, k) + self.alpha * self.get_percent_returns(index, k)**2)
        
        return variance

    def GARCHVarianceRecursive(self, last_variance : float, index : int = 256):
        return self.beta * last_variance + self.gamma * self.get_long_term_variances(index, 0) + self.alpha * (self.get_percent_returns(index, 0)**2)

class Covariance:
    def __init__(
            self,
            covariances : pd.DataFrame, X : str = None, Y : str = None) -> None:
        self.covariances : pd.DataFrame = covariances
        self.X = X
        self.Y = Y

    def __repr__(self) -> str:
        return self.covariances.__repr__()
    
    def df(self) -> pd.DataFrame:
        return self.covariances
    
    def __getitem__(self, key) -> pd.Series:
        return self.covariances[key]

    def to_csv(self, path : str):
        self.covariances.to_csv(path)

class LongTermCovariance(Covariance):
    def __init__(self,
        product_returns : pd.DataFrame,
        product_column : str,
        column_X : str,
        column_Y : str,
        long_term_covariance_format : str) -> None:

        self.long_term_covariance_format = long_term_covariance_format  

        self.column_X : str = column_X
        self.column_Y : str = column_Y

        self.X : str = column_X
        self.Y : str = column_Y

        self.product_column = product_column

        self.product_returns : pd.DataFrame = product_returns.dropna()
        

    def LongTermCovariance(self, minimum_observations : int, maximum_observations : int, date_column : str) -> float:
        if len(self.product_returns) < minimum_observations:
            raise ValueError(f"Length of DataFrame: {len(self.product_returns)} is less than {minimum_observations}.")



        first_long_term_covariance = self.product_returns[self.product_column][:minimum_observations].mean()

        dates_up_to_minimum_observations = self.product_returns.index[:minimum_observations]
        
        # create a DataFrame with all the long-term variances (duplicate the LT variance after minimum observations for all previous observations)
        df : pd.DataFrame = pd.DataFrame({self.get_LT_column_name(): [first_long_term_covariance]*(minimum_observations)}, index=dates_up_to_minimum_observations)

        long_term_covariances = []
        dates = []

        for i in range(minimum_observations, len(self.product_returns)):
            start = i - maximum_observations if i - maximum_observations > 0 else 0

            long_term_covariance = self.product_returns[self.product_column][start:i].mean()

            # appending both value and dates at same time to keep in sync
            long_term_covariances.append(long_term_covariance)
            dates.append(self.product_returns.index[i])

        df = pd.concat([df, pd.DataFrame({self.get_LT_column_name(): long_term_covariances}, index=dates)])

        # set index name
        df.index.name = date_column

        return Covariance(df)
    
    def get_LT_column_name(self) -> str:
        return self.long_term_covariance_format % (self.X, self.Y)

class GARCHCovariance(Covariance):
    def __init__(
            self,
            long_term_covariances : pd.DataFrame,
            long_term_covariance_format : str,
            product_returns : pd.DataFrame,
            product_column : pd.DataFrame,
            column_X : str,
            column_Y : str,
            weights : Tuple[float, float, float],
            GARCH_covariance_format : str) -> None:

        self.GARCH_covariance_format = GARCH_covariance_format
        
        long_term_covariance_column = long_term_covariance_format % (column_X, column_Y)

        self.long_term_covariances = long_term_covariances[long_term_covariance_column].dropna()
        
        self.column_X = column_X
        self.column_Y = column_Y

        self.X : str = column_X
        self.Y : str = column_Y

        self.product_column = product_column

        self.product_returns : pd.DataFrame = product_returns.dropna()
        
        self.alpha, self.beta, self.gamma = weights
        if not isclose(self.alpha + self.beta + self.gamma, 1.0, rel_tol=1e-9):
            raise ValueError(f"The sum of the weights: {self.alpha + self.beta + self.gamma} must equal 1.0")

    def GARCHCovariance(self, minimum_observations : int, date_column : str) -> Covariance:
        if len(self.product_returns) < minimum_observations:
            raise ValueError(f"Length of DataFrame: {len(self.product_returns)} is less than {minimum_observations}.")
        if len(self.product_returns) != len(self.long_term_covariances):
            raise ValueError(f"Length of DataFrame: {len(self.product_returns)} is not equal to the length of long term covariances: {len(self.long_term_covariances)}.")

        GARCH_covariance = self.GARCHCovarianceExplicit(minimum_observations)

        GARCH_covariances = []

        for i in range(minimum_observations, len(self.product_returns)):
            GARCH_covariance = self.GARCHCovarianceRecursive(GARCH_covariance, i)
            GARCH_covariances.append(GARCH_covariance)

        # Create NaNs for the first minimum_observations
        df = pd.DataFrame({self.get_GARCH_column_name(): [np.NaN]*minimum_observations}, index=self.product_returns.index[0:minimum_observations])

        # Set index name
        df.index.name = date_column

        # Concatenate the GARCH_covariances to the DataFrame
        df = pd.concat([df, pd.DataFrame({self.get_GARCH_column_name(): GARCH_covariances}, index=self.product_returns.index[minimum_observations:])])

        return Covariance(df)

    def get_long_term_covariances(self, index, k):
        return self.long_term_covariances[index-k]
    
    def get_product_percent_returns(self, index, k):
        return self.product_returns[self.product_column][index-k]
    
    def GARCHCovarianceExplicit(self, index : int = 256):
        covariance = 0
        for k in range(0, index-1):
            covariance += self.beta**(k) * (self.gamma * self.get_long_term_covariances(index, k) + self.alpha * self.get_product_percent_returns(index, k))
        return covariance
    
    def GARCHCovarianceRecursive(self, covariance, index : int):
        return self.beta * covariance + self.gamma * self.get_long_term_covariances(index, 0) + self.alpha * self.get_product_percent_returns(index, 0)

    def get_GARCH_column_name(self):
        return self.GARCH_covariance_format % (self.X, self.Y)

class VAR:
    def __init__(
            self,
            VARs : pd.DataFrame) -> None:
        self.VARs : pd.DataFrame = VARs

    def __repr__(self) -> str:
        return self.VARs.__repr__()

    def df(self) -> pd.DataFrame:
        return self.VARs

    def to_csv(self, path : str):
        self.VARs.to_csv(path)


#! has an asymmetric view of risk, probably worth taking the absolute value of the returns
class HistoricalVAR:
    def HistoricalVAR(
            daily_returns : pd.DataFrame, 
            VAR_column : str, 
            observation_window : int, 
            return_column : str, 
            quantile : float, 
            date_column : str) -> float:
        daily_returns.dropna(inplace=True)

        if len(daily_returns) < observation_window:
            raise ValueError(f"Length of DataFrame: {len(daily_returns)} is less than {observation_window}.")

        first_VAR = daily_returns[return_column][:observation_window].quantile(quantile)

        dates_up_to_minimum_observations = daily_returns.index[0:observation_window]
        
        # create a DataFrame with all the long-term variances (duplicate the LT variance after minimum observations for all previous observations)
        df : pd.DataFrame = pd.DataFrame({VAR_column: [first_VAR]*(observation_window)}, index=dates_up_to_minimum_observations)

        VARs = []
        dates = []

        for i in range(observation_window, len(daily_returns)):
            start = i - observation_window if i - observation_window > 0 else 0

            value_at_risk = daily_returns[return_column][start:i].quantile(quantile)

            # appending both value and dates at same time to keep in sync
            VARs.append(value_at_risk)
            dates.append(daily_returns.index[i])

        df = pd.concat([df, pd.DataFrame({VAR_column: VARs}, index=dates)])

        # set index name
        df.index.name = date_column

        return VAR(df)
