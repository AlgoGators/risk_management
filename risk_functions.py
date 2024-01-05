"""Contains Functions Relating to Risk Assessment"""

from statistics import NormalDist
from math import sqrt
import numpy as np
import pandas as pd
from enum import Enum

# Ugly but it allows keeping the same import statement across submodules and parent directories
try:
    from .constants import BUSINESS_DAYS_IN_YEAR
    from . import statistical_functions as StatisticalCalculations
    from .statistical_functions import Periods
except ImportError:
    from constants import BUSINESS_DAYS_IN_YEAR
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
            - No changing in position size over given period
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
            period (Days): the period for which we are calculating the max downside risk
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
            IDM, 
            instrument_weight, 
            risk_target, 
            annualized_stddev,
            average_forecast,
            max_forecast,
            max_leverage_ratio,
            capital,
            notional_exposure_per_contract,
            open_interest,
            max_pct_of_open_interest,
            max_forecast_margin=0.5) -> float:
        """Returns the lesser of the max position based on forecast, leverage, and open interest"""

        return min(
            self.maximum_position_forecast(number_of_contracts, capital, IDM, instrument_weight, risk_target, notional_exposure_per_contract, annualized_stddev, average_forecast, max_forecast, max_forecast_margin=max_forecast_margin), 
            self.maximum_position_leverage(number_of_contracts, max_leverage_ratio, capital, notional_exposure_per_contract), 
            self.maximum_position_open_interest(number_of_contracts, open_interest, max_pct_of_open_interest))

    def maximum_position_forecast(
            self,
            number_of_contracts : float,
            capital : float,
            IDM : float,
            instrument_weight : float,
            risk_target : float,
            notional_exposure_per_contract : float,
            stddev : float,
            average_forecast : int,
            max_forecast : int,
            max_forecast_margin : float = 0.50) -> float:
        """
        Determines maximum position based on maximum forecast
        
        Parameters:
        ---
            number_of_contracts : float
                number of contracts forecasted by the algorithm
            scaled_forecast : float
                the scaled forecast created by the algorithm
            average_forecast : int
                average absolute forecast (Carver uses 10)
            max_forecast : int
                maximum absolute forecast (Carver uses 20)
            max_forecast_margin : float
                the margin around the max forecasted position acceptable 
                (Carver uses 50% so the position can exceed the max forecast by 50%)
        ---
        """

        #@                             zmax_forecast * capital * IDM * instrument_weight * risk_target
        #@ max_position_forecast   =   --------------------------------------------------------
        #@                             average_forecast * notional_exposure_per_contract * stddev

        max_forecast_ratio = max_forecast / average_forecast

        max_position_forecast = max_forecast_ratio * (capital * IDM * instrument_weight * risk_target) / (notional_exposure_per_contract * stddev)

        maximum_position = max_position_forecast * (1 + max_forecast_margin)

        return min(maximum_position, number_of_contracts)

    def maximum_position_leverage(
            self,
            number_of_contracts : float,
            max_leverage_ratio : float,
            capital : float,
            notional_exposure_per_contract : float) -> float:
        """
        Determines maximum position relative to maximum leverage
        
        Parameters:
        ---
            number_of_contracts : float
                number of contracts forecasted by the algorithm
            max_leverage_ratio : float
                the greatest level of leverage acceptable for any single position
            capital : float
                the amount of capital expected to be allocated to this position
            notional_exposure_per_contract : float 
                equals the multiplier * price * fx_rate
        ---
        """

        max_leverage = max_leverage_ratio * capital / notional_exposure_per_contract

        return min(number_of_contracts, max_leverage)

    def maximum_position_open_interest(
            self,
            number_of_contracts : float,
            open_interest : int,
            max_pct_of_open_interest : float = 0.01) -> float:
        """
        Determines maximum acceptable position in order to not exceed a certain % of open interest
        
        Parameters:
        ---
            number_of_contracts : float
                number of contracts forecasted by the algorithm
            open_interest : int
                the open interest for a given instrument
            max_open_interest : float
                the max acceptable % of open interest a position could be
        ---

        """

        return min(number_of_contracts, open_interest * max_pct_of_open_interest)


class Volatility():
    # TODO: add calculations, change default value for minimum volatility
    def minimum_volatility(
            self,
            IDM : float,
            instrument_weight : float,
            risk_target : float,
            instrument_returns : list[float],
            maximum_leverage : float) -> bool:
        """
        Returns true if the returns for a given instrument meets a minimum level of volatility, false if not
        
        Parameters:
        ---
            IDM : float
                instrument diversification multiplier
            instrument_weight : float
                the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            risk_target : float
                the target risk for the portfolio
            instrument_returns : list[float]
                the returns for a given instrument
            maximum_leverage : float
                the max acceptable leverage for a given instrument
        ---
        """

        minimum_volatility = (2 * IDM * instrument_weight * risk_target) / maximum_leverage

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
            position_weights : DataFrame
                columns are the weight for each instrument, use get_position_weight to get each weight
            position_percent_returns : DataFrame
                each column are % returns for each ticker
            max_portfolio_risk : float
                max risk for the portfolio (should technically be 99th percentile of annualized risk)
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
        
        """
        Parameters:
        ---
            position_weights : pd.DataFrame
                each column should have the position weight for each ticker
                NOTE this is different from instrument weights:
                this weight is the notional exposure of the instrument divided by the total portfolio capital
        ---
        
        """
        
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