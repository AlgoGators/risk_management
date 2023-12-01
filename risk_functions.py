"""Contains Functions Relating to Risk Assessment"""

from statistics import NormalDist
from math import sqrt
import numpy as np
import pandas as pd
from constants import BUSINESS_DAYS_IN_YEAR, BUSINESS_DAYS_IN_TEN_YEARS
from enum import Enum
import statistical_functions as StatisticalCalculations
from statistical_functions import Periods


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
    def maximum_position(
            self,
            number_of_contracts, 
            open_interest, 
            pct_of_interest,
            max_leverage_ratio, 
            max_forecast, 
            average_forecast, 
            annual_stddev,
            IDM,
            weight,
            risk_target,
            capital,
            multiplier,
            price, 
            fx_rate=1.0,
            max_forecast_margin=0.5) -> float:
        """Returns the lesser of the max position based on forecast, leverage, and open interest"""

        return min(
            self.maximum_position_forecast(number_of_contracts, max_forecast, average_forecast, IDM, weight, risk_target, multiplier, price, annual_stddev, capital, fx_rate, max_forecast_margin), 
            self.maximum_position_leverage(number_of_contracts, max_leverage_ratio, capital, multiplier, price, fx_rate), 
            self.maximum_position_open_interest(number_of_contracts, open_interest, pct_of_interest))

    def maximum_position_forecast(
            self,
            number_of_contracts : float,
            max_forecast : int,
            average_forecast : int,
            IDM : float,
            weight : float,
            risk_target : float,
            multiplier : float,
            price : float,
            annual_stddev : float,
            capital : float,
            fx_rate : float = 1.0,
            max_forecast_margin : float = 0.50) -> float:
        """Determines maximum position based on maximum forecast"""

        maximum_position = ((max_forecast / average_forecast) * capital * IDM * weight * risk_target / (multiplier * price * fx_rate * annual_stddev)) * (1 + max_forecast_margin)

        return min(maximum_position, number_of_contracts)

    def maximum_position_leverage(
            self,
            number_of_contracts : float,
            max_leverage_ratio : float,
            capital : float,
            multiplier : float,
            price : float,
            fx_rate : float = 1.0) -> float:
        """Determines maximum position relative to maximum leverage"""

        return min(number_of_contracts, (max_leverage_ratio * capital) / (multiplier * price * fx_rate))

    def maximum_position_open_interest(
            self,
            number_of_contracts : float,
            open_interest : int,
            pct_of_interest : float) -> float:
        """Determines maximum positions as a fraction of open interest"""

        return min(number_of_contracts, open_interest * pct_of_interest)


class Volatility():
    # TODO: add calculations, change default value for minimum volatility
    def minimum_volatility(
            self,
            IDM : float,
            weight : float,
            risk_target : float,
            instrument_returns : list[float], 
            maximum_leverage : float) -> bool:
        """Returns true if the returns for a given instrument meets a minimum level of volatility, false if not"""

        minimum_volatility = (2 * IDM * weight * risk_target) / maximum_leverage

        standard_deviation = StatisticalCalculations.std(instrument_returns, annualize=True)

        if (standard_deviation < minimum_volatility):
            return False        

        return True


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
        
        portfolio_standard_deviation = StatisticalCalculations.portfolio_stddev(position_weights, position_percent_returns)

        return min(1, max_portfolio_risk / portfolio_standard_deviation)


    def jump_risk_multiplier(
            self,
            position_weights : pd.DataFrame,
            position_percent_returns : pd.DataFrame,
            max_portfolio_risk : float = 0.70) -> float:

        stddev_lst = []

        tickers = position_percent_returns.columns.tolist()

        for ticker in tickers:
            rolling_stddevs = StatisticalCalculations.rolling_std(position_percent_returns[ticker])
            stddev_lst.append(np.percentile(rolling_stddevs, 99))


        stddev_array = np.array(stddev_lst)

        corr_matrix = StatisticalCalculations.correlation_matrix(position_percent_returns, Periods.WEEKLY, 52)

        covariance_matrix = np.dot(np.dot(np.diag(stddev_array), corr_matrix), np.diag(stddev_array))

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

        tickers = position_percent_returns.columns.tolist()

        risk_lst = []

        for ticker in tickers:
            # get the most recent value
            rolling_stddev = StatisticalCalculations.rolling_std(position_percent_returns[ticker])[-1]
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