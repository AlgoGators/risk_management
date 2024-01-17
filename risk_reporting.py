import numpy as np
from math import sqrt
from constants import BUSINESS_DAYS_IN_YEAR
from statistics import NormalDist

try:
    from .constants import BUSINESS_DAYS_IN_YEAR
except ImportError:
    from constants import BUSINESS_DAYS_IN_YEAR

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
