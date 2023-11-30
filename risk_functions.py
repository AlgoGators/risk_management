"""Contains Functions Relating to Risk Assessment"""

from statistics import NormalDist
from math import sqrt
import numpy as np
import pandas as pd
from constants import BUSINESS_DAYS_IN_YEAR, BUSINESS_DAYS_IN_TEN_YEARS
from enum import Enum, auto
# from risk_constants import MarginLevels


class Periods(Enum):
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 20
    QUARTERLY = 60
    YEARLY = 256


class StatisticalCalculations():
    def SMA(
            self, 
            lst : list[float], 
            span : int = -1
        ) -> float:

        if (span == -1):
            return np.nanmean(lst)
        
        return np.nanmean(lst[-span])


    def SR(self, returns : list[float]) -> float:
        mean = np.nanmean(returns)
        stddev = self.stddev(returns)

        return mean / stddev


    def EWMA (
            self, 
            lst : list[float], 
            span : int = None,
            alpha : float = None,
            threshold : int = 100) -> float:
        """Returns Exponentially Weighted Moving Average given span"""

        #@ EWMA(t) = α(1 - α)⁰ * y(t) + α(1 - α)¹ * y(t-1) + α(1-α)² * y(t-2) + ... + α(1-α)ⁿ * y(t-n)
        # where alpha is 2 / (span + 1) & n is the length of the list

        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")
        

        if alpha is None:
            alpha : float = 2 / (span + 1)


        lst_len : int = len(lst)
        last_IDX : int = lst_len - 1

        if (lst_len <= threshold):
            return self.SMA(lst)
        
        ewma : float = self.SMA(lst[:threshold])


        for n in range(threshold, lst_len):
            ewma += (alpha * (1-alpha)**n * lst[last_IDX - n])

        return ewma


    def stddev(
            self,
            lst : list[float],
            span : int = -1,
            annualize : bool = False) -> float:
        
        """Returns the standard deviation of a list of values"""

        # get the lesser of the two values
        span = min(len(lst), span)

        reduced_lst : list[float] = lst[:-span] if (span != -1) else lst

        xbar : float = np.mean(reduced_lst)

        numerator : float = 0

        for x in reduced_lst:
            numerator += (x - xbar)**2

        lst_len : int = len(reduced_lst)

        if lst_len == 1:
            return 0

        standard_deviation : float = sqrt(numerator / (len(reduced_lst) - 1))

        factor = 1

        if (annualize is True):
            factor = sqrt(BUSINESS_DAYS_IN_YEAR)

        return standard_deviation * factor 
    

    def VAR(
        self,
        lst : list[float],
        span : int = -1,
        annualize : bool = False) -> float:

        return self.stddev(lst=lst, span=span, annualize=annualize)**2
    

    def exponentially_weighted_stddev(
            self,
            lst : list,
            span : int = None, 
            alpha : float = None, 
            annualize : bool = False,
            threshold : int = 100) -> float:
        """
        """

        #@ given an expoentially weighted moving average, r*
        #@                     _________________________________________________________________________________
        #@ exponential σ(t) = √ α(1 - α)⁰(r(t) - r*)² + α(1 - α)¹(r(t-1) - r*)² + α(1 - α)²(r(t-2) - r*)² + ... 


        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")

        ewma : float = self.EWMA(lst, span=span, alpha=alpha, threshold=threshold)
        
        if alpha is None:
            alpha : float = 2 / (span + 1)

        radicand : float  = 0
        lst_len : int = len(lst)
        last_IDX : int = lst_len - 1

        if (lst_len <= threshold):
            return self.stddev(lst=lst, span=span, annualize=annualize)

        # starting value is just the simple variance of the first 100 values (threshold). Variance is the radicand of the stddev formula
        radicand = self.VAR(lst[:threshold])

        for n in range(threshold, lst_len):
            radicand += (alpha * (1 - alpha)**n * (lst[last_IDX - n] - ewma)**2) 

        ew_stddev : float = sqrt(radicand)

        factor = 1
        if (annualize is True):
            factor = sqrt(BUSINESS_DAYS_IN_YEAR)

        return ew_stddev * factor


    def exponentially_weighted_var(
        self,
        lst : list[float],
        alpha : float = None,
        span : int = None,
        threshold : int = 10) -> float:

        """
            threshold : when the formula switches from a simple variance to EW_var
        """

        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")
        
        if alpha is None:
            alpha : float = 2 / (span + 1)

        lst_len : int = len(lst)


        # if the list length is less than threshold just return the simple variance (stddev squared)
        if lst_len <= threshold:
            return self.VAR(lst=lst)
        

        # EWV(t) = λ * (x(t) − EWMA(t−1))^2 + (1 − λ) * EWV(t−1)
        # starting value is just the simple variance of the first 10 values
        # EW_var = self.stddev(lst[:10])**2

        # starting with n = 10 (threshold)
        # EW_var = alpha * (lst[n] - self.EWMA(lst[:n], alpha=alpha))**2 + (1-alpha) * EW_var
     
        EW_var : float = self.VAR(lst[:threshold])

        for n in range(threshold, lst_len):
            EW_var = alpha * (lst[n] - self.EWMA(lst[:n], alpha=alpha))**2 + (1 - alpha) * EW_var

        return EW_var


    def exponentially_weighted_covar(
        self,
        lst_X : list[float],
        lst_Y : list[float],
        alpha : float = None,
        span : int = None) -> float:
        """Calculates an exponentially weighted covariance with weight alpha"""

        #@ EW Cov(X, Y) = sum(0, t) (1 - α)^n * (X(t-1) - EWMA(X)) * (Y(t-1) - EWMA(Y))

        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")
        
        if alpha is None:
            alpha : float = 2 / (span + 1)

        EW_covar : float = 0

        lst_len : int = len(lst_X)
        last_IDX : int = lst_len - 1

        ewma_X : float = self.EWMA(lst_X, alpha=alpha)
        ewma_Y : float = self.EWMA(lst_Y, alpha=alpha)

        for n in range(0, lst_len):
            EW_covar += (1-alpha)**n * (lst_X[last_IDX-n] - ewma_X) * (lst_Y[last_IDX-n] - ewma_Y)

        return EW_covar


    def correlation(
        self,
        returns_X : pd.DataFrame,
        returns_Y : pd.DataFrame) -> float:
        """Calculates a correlation between two DataFrames where each dataframe had a "Date" column"""

        correlation = 0.0

        # Try to merge the two dataframes on the date column
        try:
            merged_df = pd.merge(returns_X, returns_Y, on="Date", how="inner")
            correlation = merged_df.iloc[:,1].corr(merged_df.iloc[:,2])
            
        # If not just merge them on the index
        except KeyError:
            merged_df = pd.merge(returns_X, returns_Y, left_index=True, right_index=True, how="inner")
            correlation = merged_df.iloc[:,0].corr(merged_df.iloc[:,1])
            
        return correlation
    

    def exponentially_weighted_correlation(
        self,
        returns_X : pd.DataFrame,
        returns_Y : pd.DataFrame,
        span : int) -> float:
        
        covar : float = StatisticalCalculations().exponentially_weighted_covar(returns_X, returns_Y, span=span)

        var_tickerX : float = StatisticalCalculations().exponentially_weighted_var(returns_X, span=span)
        var_tickerY : float = StatisticalCalculations().exponentially_weighted_var(returns_Y, span=span)


        # default value for correlation should be nan just in case variance is 0
        correlation : float = np.nan 
        
        # make sure neither variance is 0, if so then calculate the correlation
        if (0 not in [var_tickerX, var_tickerY]):
            correlation = covar / (sqrt(var_tickerX * var_tickerY))

        return correlation


    def correlation_matrix(
        self,
        returns_df : pd.DataFrame,
        period : Periods,
        window : int) -> np.array:

        #@ correlation formula
        #@ r =             EW Cov(X,Y)
        #@     -------------------------------
        #@      ______________________________
        #@     √    EW Var(X) * EW Var(Y)

        periodic_returns_df = pd.DataFrame()
        
        tickers = returns_df.columns.tolist()

        for ticker in tickers:
            returns = returns_df[ticker].tolist()

            # groups them and takes the recent window backwards
            periodic_returns_df[ticker] = [sum(returns[x : x + period.value])
                                    for x in range(0, len(returns), period.value)][:-window]
    
        correlation_matrix : list[float] = []

        for n, tickerX in enumerate(tickers):
            # always start with 1.0 in the correlations since X1 is perfectly correlated with X2
            correlations : list[float] = [1.0]

            # go through the remaining variables, ignoring those previously calculated
            for tickerY in tickers[n+1:]:
                correlation = self.correlation(periodic_returns_df[tickerX], periodic_returns_df[tickerY])

                correlations.append(correlation)

            correlation_matrix.append(correlations)

        #? right now we these correlations
        #? coordinates are relative to the existing values
        #?     0   1   2   3   4
        #?     A   B   C   D   E
        #? 0 A 1.0 0,1 0,2 0,3   
        #? 1 B 0.5 1.0 1,1 1,2
        #? 2 C 0.6 0.7 1.0 2,1
        #? 3 D 0.4 0.9 0.4 1.0
        #? 4 E 0.8 0.7 0.9 0.4 1.0

        correlation_df = pd.DataFrame(columns=tickers)

        for n, column in enumerate(correlation_matrix):
            # we need the second value in each column to the left 

            current_ticker : str = tickers[n]

            if (n == 0):
                correlation_df[current_ticker] = column
                continue

            new_column = []

            row : int = n

            for x in range(0,n):
                new_column.append(correlation_matrix[x][row])
                row -= 1

            new_column.extend(column)

            correlation_df[current_ticker] = new_column

        return np.array(correlation_df)
    

    def exponentially_weighted_correlation_matrix(
        self,
        returns_df : pd.DataFrame,
        period : Periods,
        span : int) -> np.array:

        #@ correlation formula
        #@ r =             EW Cov(X,Y)
        #@     -------------------------------
        #@      ______________________________
        #@     √    EW Var(X) * EW Var(Y)

        periodic_returns_df = pd.DataFrame()
        
        tickers = returns_df.columns.tolist()

        for ticker in tickers:
            returns = returns_df[ticker].tolist()

            periodic_returns_df[ticker] = [sum(returns[x : x + period.value])
                                    for x in range(0, len(returns), period.value)]
    
        correlation_matrix : list[float] = []

        for n, tickerX in enumerate(tickers):
            # always start with 1.0 in the correlations since X1 is perfectly correlated with X2
            correlations : list[float] = [1.0]

            # go through the remaining variables, ignoring those previously calculated
            for tickerY in tickers[n+1:]:
                correlation = self.exponentially_weighted_correlation(periodic_returns_df[tickerX], periodic_returns_df[tickerY], span=span)

                correlations.append(correlation)

            correlation_matrix.append(correlations)

        #? right now we these correlations
        #? coordinates are relative to the existing values
        #?     0   1   2   3   4
        #?     A   B   C   D   E
        #? 0 A 1.0 0,1 0,2 0,3   
        #? 1 B 0.5 1.0 1,1 1,2
        #? 2 C 0.6 0.7 1.0 2,1
        #? 3 D 0.4 0.9 0.4 1.0
        #? 4 E 0.8 0.7 0.9 0.4 1.0

        correlation_df = pd.DataFrame(columns=tickers)

        for n, column in enumerate(correlation_matrix):
            # we need the second value in each column to the left 

            current_ticker : str = tickers[n]

            if (n == 0):
                correlation_df[current_ticker] = column
                continue

            new_column = []

            row : int = n

            for x in range(0,n):
                new_column.append(correlation_matrix[x][row])
                row -= 1

            new_column.extend(column)

            correlation_df[current_ticker] = new_column

        return np.array(correlation_df)
    

    def rolling_stddev(
        self,
        returns : pd.DataFrame,
        ten_year_weight : float = 0.3) -> float:
        """Calculates a rolling standard deviation for a given dataframe with weighting on the annualized stddev and 10 year average"""

        annualized_stddevs = []
        ten_year_averages = []

        # max values included in ew_stddev, this should expedite the process
        maximum_values = 100
        
        weighted_stddevs = []

        for n, val in enumerate(returns.tolist()):
            start = max(0, n - maximum_values)
            annualized_stddev = self.exponentially_weighted_stddev(returns[start:n+1], span=32, annualize=True)

            annualized_stddevs.append(annualized_stddev)

            if n < BUSINESS_DAYS_IN_TEN_YEARS:
                ten_year_average = np.mean(annualized_stddevs[:n+1])
                
            else:
                ten_year_average =np.mean(annualized_stddevs[n-BUSINESS_DAYS_IN_TEN_YEARS:n+1])

            ten_year_averages.append(ten_year_average)

            weighted_stddev = ten_year_weight * ten_year_average + (1 - ten_year_weight) * annualized_stddev

            weighted_stddevs.append(weighted_stddev)

        return weighted_stddevs


    def portfolio_covar(
            self, 
            position_percent_returns : pd.DataFrame) -> np.array:
        """Calculates a covariance matrix as outlined by Carver on pages 606-607"""

        #@ Σ = σ.ρ.σᵀ = σσᵀ ⊙ ρ (using Hadamard product) = Diag(σ) * ρ * Diag(σ)
        #@ where:
        #@ ρ is the correlation matrix
        #@ σ is the vector of annualized estimates of % standard deviations 
        #@ use 32 day span for standard deviations
        #@ window for equally weighted correlation matrix of 52 weeks


        stddev_lst = []

        tickers = position_percent_returns.columns.tolist()

        for ticker in tickers:
            # get the most recent value
            rolling_stddev = self.rolling_stddev(position_percent_returns[ticker])[-1]
            stddev_lst.append(rolling_stddev)

        stddev_array = np.array(stddev_lst)

        correlation_matrix = StatisticalCalculations().correlation_matrix(position_percent_returns, Periods.WEEKLY, 52)

        EW_covar = np.dot(np.dot(np.diag(stddev_array), correlation_matrix), np.diag(stddev_array))

        return EW_covar

    
    def portfolio_stddev(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame) -> float:
        
        #@                _______
        #@ Portfolio σ = √ w Σ wᵀ
        #@ w is the vector of positions weights, and Σ is the covariance matrix of percent returns 

        tickers : list = position_weights.columns.tolist()

        weights_lst : list = []

        # gets the weights for each instrument
        for ticker in tickers:
            weights_lst.append(position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

        weights = np.array(weights_lst)

        weights_T = weights.transpose()

        covariance_matrix = StatisticalCalculations().portfolio_covar(position_percent_returns)

        radicand : float = np.dot(np.dot(weights, covariance_matrix), weights_T)

        return sqrt(radicand)
    


# TODO
def leverage_ratio(portfolio):
    """Takes Portfolio Data and Calculates the Leverage for all Assets"""

    # TODO Add some alert if outside of 1.5x - 3x


class RiskEstimates():
    def VaR_Historical(
            position_value : float,
            historical_returns : list[float],
            alpha : float = 0.05) -> float:
        """
        Gets VaR, where only α% of theoretical losses will be greater

        Assumptions:
        -----
            - No changing in positions over a certain period
            - Past returns ARE indicative of future results

        -----

        Parameters:
        -----
            position_value : a dollar value of the investment(s)
            historical_returns : list of percentage returns
            alpha : value from (0, 1) exclusive. e.g. 0.1, 0.05, 0.01
                smaller alpha (i.e. greater level of confidence) => greater max downside
        -----
        """

        nth_percentile_percent = np.percentile(historical_returns, alpha)

        VaR = abs(nth_percentile_percent * position_value)

        return VaR

    def VaR_Parametric(
            position_value : float,
            expected_volatility : float,
            period : int,
            confidence_level : float = 0.05) -> float:
        """
        Gets VaR, where α% of all possible losses will be smaller

        Assumptions:
        -----
            - No changing in positions over a certain period
            - Future returns ARE normally distributed

        -----

        Parameters:
        -----
            method : a choice of VaRMethod.PARAMETRIC or VaRMethod.HISTORICAL
            position_value : a dollar value of the investment(s)
            expected_volatility : expected volatility for the investment value
                greater volatility => greater max downside
            period : the period for which we are calculating the max downside risk
                greater period => greater max downside
            alpha : value from (0, 1) exclusive. e.g. 0.1, 0.05, 0.01
                smaller alpha (i.e. greater level of confidence) => greater max downside
        -----
        """

        #* Formula for Parametric Method:
        #@ VaR = Investment Value * Z-score * Expected Volatility * sqrt (Time Horizon / trading days)
        z_score = NormalDist().inv_cdf(confidence_level / 100)

        VaR = abs(position_value * z_score * expected_volatility * sqrt(period / BUSINESS_DAYS_IN_YEAR))

        return VaR


class PositionLimits():
    def maximum_position(self):
        """Returns the lesser of the max position based on forecast, leverage, and open interest"""
        return min(self.maximum_position_forecast(), self.maximum_position_leverage(), self.maximum_position_open_interest())

    def maximum_position_forecast(self) -> float:
        """Determines maximum position based on maximum forecast"""
        maximum_position = 0.0

        return maximum_position

    def maximum_position_leverage(self) -> float:
        """Determines maximum position relative to maximum leverage"""
        maximum_position = 0.0

        return maximum_position

    def maximum_position_open_interest(self) -> float:
        """Determines maximum positions as a fraction of open interest"""
        maximum_position = 0.0

        return maximum_position


class Volatility():
    # TODO: add calculations, change default value for minimum volatility
    def minimum_volatility(
            instrument_returns : list[float], 
            minimum_volatility : float) -> bool:
        """Returns true if the returns for a given instrument meets a minimum level of volatility, false if not"""


class RiskOverlay():
    def estimated_portfolio_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame,
            max_portfolio_risk : float = 0.30) -> float:
        """
        Parameters:
        -----
            position_weights : DataFrame, columns are the weight for each instrument
            position_percent_returns : DataFrame, each column are % returns for each ticker
            max_portfolio_risk : max risk for the portfolio (should technically be 99th percentile of annualized risk)
        -----
        
        """
        
        portfolio_standard_deviation = StatisticalCalculations().portfolio_stddev(position_weights, position_percent_returns)

        return min(1, max_portfolio_risk / portfolio_standard_deviation)


    def jump_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame,
            max_portfolio_risk : float = 0.70) -> float:

        stddev_lst = []

        tickers = position_percent_returns.columns.tolist()

        for ticker in tickers:
            rolling_stddevs = StatisticalCalculations().rolling_stddev(position_percent_returns[ticker])
            stddev_lst.append(np.percentile(rolling_stddevs, 99))


        stddev_array = np.array(stddev_lst)

        correlation_matrix = StatisticalCalculations().correlation_matrix(position_percent_returns, Periods.WEEKLY, 52)

        covariance_matrix = np.dot(np.dot(np.diag(stddev_array), correlation_matrix), np.diag(stddev_array))

        weights_lst : list = []

        # gets the weights for each instrument
        for ticker in tickers:
            weights_lst.append(position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

        weights = np.array(weights_lst)

        weights_T = weights.transpose()

        radicand : float = np.dot(np.dot(weights, covariance_matrix), weights_T)

        return min(1, max_portfolio_risk / sqrt(radicand))
    

    def correlation_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame,
            max_portfolio_risk : float = 0.65) -> float:

        stddev_lst = []

        tickers = position_percent_returns.columns.tolist()

        risk_lst = []

        for ticker in tickers:
            # get the most recent value
            rolling_stddev = StatisticalCalculations().rolling_stddev(position_percent_returns[ticker])[-1]
            risk_lst.append(rolling_stddev * position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

        risk_lst = np.array(risk_lst)

        risk_lst = np.absolute(risk_lst)

        standard_deviation = np.sum(risk_lst)

        return min(1, max_portfolio_risk / standard_deviation)
    

    def leverage_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            max_portfolio_leverage : float = 20) -> float:
        
        tickers = position_weights.columns.tolist()

        weights_lst : list = []

        # gets the weights for each instrument
        for ticker in tickers:
            weights_lst.append(position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

        weights = np.array(weights_lst)

        absolute_weights = np.absolute(weights)

        leverage = np.sum(absolute_weights)

        return min(1, max_portfolio_leverage, leverage)


    def final_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame) -> float:
        
        return min(self.estimated_portfolio_risk_multiplier(position_weights, position_percent_returns), self.jump_risk_multiplier(position_weights, position_percent_returns), self.correlation_risk_multiplier(position_weights, position_percent_returns), self.leverage_risk_multiplier(position_weights))

class MarginLevels(float, Enum):
    """Margin Level Concern and Upper Limit"""
    NO_ACTION = 0.0
    ALERT = 0.50
    TAKE_ACTION = 0.60


class Margins():
    # TODO: Create function to determine margin of a given portfolio
    def margin_level(self, portfolio) -> float:
        margin : float = 0.75

        return margin

    # TODO: Create scaling reduction function
    def scale_portfolio(self, portfolio, scaleback : float):
        """Returns a portfolio that has been reduced to the scaleback fraction of the original value"""

        return portfolio

    # TODO 
    def margin_action(self, portfolio):
        margin = self.margin_level(portfolio)

        if (margin < MarginLevels.ALERT):
            return MarginLevels.NO_ACTION
        
        if (margin < MarginLevels.TAKE_ACTION):
            return MarginLevels.ALERT
        
        # at this point the Margin Level is high enough to require scaling back to the ALERT level
        scaleback : float = MarginLevels.ALERT / margin

        return self.scale_portfolio(portfolio, scaleback)



if __name__ == "__main__":
    pass