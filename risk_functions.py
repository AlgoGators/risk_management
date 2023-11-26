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
            alpha : float = None) -> float:
        """Returns Exponentially Weighted Moving Average given span"""

        #@ EWMA(t) = α(1 - α)⁰ * y(t) + α(1 - α)¹ * y(t-1) + α(1-α)² * y(t-2) + ... + α(1-α)ⁿ * y(t-n)
        # where alpha is 2 / (span + 1) & n is the length of the list

        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")
        
        if alpha is None:
            alpha : float = 2 / (span + 1)


        ewma : float = 0
        lst_len : int = len(lst)
        last_IDX : int = lst_len - 1

        for n in range(0, lst_len):
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

        standard_deviation : float = sqrt(numerator / (len(reduced_lst) - 1))

        factor = 1

        if (annualize is True):
            factor = sqrt(BUSINESS_DAYS_IN_YEAR)

        return standard_deviation * factor 
    

    def exponentially_weighted_stddev(
            self,
            lst : np.array,
            span : int = None, 
            alpha : float = None, 
            annualize : bool = False) -> float:
        """
        """

        #@ given an expoentially weighted moving average, r*
        #@                     _________________________________________________________________________________
        #@ exponential σ(t) = √ α(1 - α)⁰(r(t) - r*)² + α(1 - α)¹(r(t-1) - r*)² + α(1 - α)²(r(t-2) - r*)² + ... 


        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")

        ewma : float = self.EWMA(lst, span=span, alpha=alpha)
        
        if alpha is None:
            alpha : float = 2 / (span + 1)

        radicand : float  = 0
        lst_len : int = len(lst)
        last_IDX : int = lst_len - 1

        for n in range(0, lst_len):
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
        span : int = None) -> float:

        # checks that only one of the variables is given
        if not(any([span, alpha]) and not all([span, alpha])):
            raise ValueError("Only one of span or alpha may be used")
        
        ewma : float = self.EWMA(lst, span=span, alpha=alpha)
        
        if alpha is None:
            alpha : float = 2 / (span + 1)


        EW_var : float = 0

        lst_len : int = len(lst)
        last_IDX : int = lst_len - 1

        for n in range(0, lst_len):
            EW_var += (1 - alpha)**n * (lst[last_IDX - n] - ewma)**2

        return EW_var


    def exponentially_weighted_covar(
        self,
        lst_X : list[float],
        lst_Y : list[float],
        ewma_X : float,
        ewma_Y : float,
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

        for n in range(0, lst_len):
            EW_covar += (1-alpha)**n * (lst_X[last_IDX-n] - ewma_X) * (lst_Y[last_IDX-n] - ewma_Y)

        return EW_covar


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
                ewma_tickerX : float = StatisticalCalculations().EWMA(periodic_returns_df[tickerX], span=span)
                ewma_tickerY : float = StatisticalCalculations().EWMA(periodic_returns_df[tickerY], span=span)

                covar : float = StatisticalCalculations().exponentially_weighted_covar(periodic_returns_df[tickerX], periodic_returns_df[tickerY], ewma_tickerX, ewma_tickerY, span=span)

                var_tickerX : float = StatisticalCalculations().exponentially_weighted_var(periodic_returns_df[tickerX], ewma_tickerX, span=span)
                var_tickerY : float = StatisticalCalculations().exponentially_weighted_var(periodic_returns_df[tickerY], ewma_tickerY, span=span)


                # default value for correlation should be nan just in case variance is 0
                correlation : float = np.nan 
                
                # make sure neither variance is 0, if so then calculate the correlation
                if (0 not in [var_tickerX, var_tickerY]):
                    correlation = covar / (sqrt(var_tickerX * var_tickerY))

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
    

    def exponentially_weigthed_portfolio_covar(
            self, 
            position_percent_returns : pd.DataFrame) -> np.array:
        """Calculates a covariance matrix as outlined by Carver on pages 606-607"""

        #@ Σ = σ.ρ.σᵀ
        #@ where:
        #@ ρ is the correlation matrix
        #@ σ is the vector of annualized estimates of % standard deviations 
        #@ use 32 day span for standard deviations
        #@ period for exponentially weighted correlation matrix of 25 weeks


        stddev_lst = []

        tickers = position_percent_returns.columns.tolist()

        for ticker in tickers:
            stddev_lst.append(StatisticalCalculations().stddev(position_percent_returns[ticker], 32, True))

        stddev_array = np.array(stddev_lst)

        correlation_matrix = StatisticalCalculations().exponentially_weighted_correlation_matrix(position_percent_returns, Periods.WEEKLY, 25)

        stddev_array_T = np.transpose(stddev_array)

        EW_covar = np.dot(np.dot(stddev_array, correlation_matrix), stddev_array_T)

        return EW_covar


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
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame,
            expected_annualized_percent_stddev : np.array) -> float:
        """
        Parameters:
        -----
            position_weights : DataFrame, columns are the weight for each instrument
            position_percent_returns : DataFrame, each column are % returns for each ticker
            expected_annualized_percent_stddev : np.array, expected annualized percent stddevs
        -----
        
        """
        

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



        expected_annualized_percent_stddevs = []
        
        for ticker in tickers:
            # carver recommends a 32 day span
            expected_annualized_percent_stddevs.append(StatisticalCalculations().exponentially_weighted_stddev(np.array(position_percent_returns[ticker]), span=32, annualize=True))



        # ensure the position_percent_returns are in the same order as tickers
        position_percent_returns = position_percent_returns.reindex(columns=tickers)

        covariance_matrix_df = position_percent_returns.cov()

        covariance_matrix_array = covariance_matrix_df.to_numpy()

        radicand : float = np.dot(np.dot(weights, covariance_matrix_array), weights_T)

        portfolio_standard_deviation = sqrt(radicand)

        percentile_99th = np.percentile(expected_annualized_percent_stddev, 99)

        return min(1, percentile_99th / portfolio_standard_deviation)


    def jump_risk_multiplier() -> float:
        pass

    def correlation_risk_multiplier() -> float:
        pass

    def leverage_risk_multiplier() -> float:
        pass


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