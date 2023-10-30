"""Contains Functions Relating to Risk Assessment"""

from enum import Enum, auto
from statistics import NormalDist
from math import sqrt
from constants import *
import pandas as pd
import numpy as np


class VaRMethod(Enum):
    """List of Value at Risk(VaR) Methods"""
    PARAMETRIC = auto()
    HISTORICAL = auto()


def get_value_at_risk(
        method : VaRMethod, 
        investment_value : float, 
        confidence_level : int,
        historical_returns : list[float] = None,
        expected_volatility : float = None,
        period : int = None) -> float:
    """
    Gets VaR, where x% of all possible losses will be smaller

    Assumptions:
    -----
        - No changing in positions over a certain period

        - Either:
            - Past returns ARE indicative of future results (for Historical)
            - Future returns ARE normally distributed (for Parametric)

    -----

    Parameters:
    -----
        method : a choice of VaRMethod.PARAMETRIC or VaRMethod.HISTORICAL
        investment_value : a dollar value of the investment(s)
        confidence_level : value from (0, 100) exclusive. e.g. 90, 95, 99
            greater levels of confidence => greater max downside
        historical_returns : list of percentage returns for a Historical Simulation VaR
        expected_volatility : expected volatility for the investment value for a Parametric VaR
            greater volatility => greater max downside
        period : the period for which we are calculating the max downside risk for a Parametric VaR
            greater period => greater max downside
    -----
    """

    if (method.name not in VaRMethod._member_names_):
        raise TypeError(f"An inappropriate {VaRMethod.__name__} was provided, please select from: {VaRMethod._member_names_}")
    

    if (method == VaRMethod.PARAMETRIC):
        if (period is None or expected_volatility is None):
            raise TypeError(
                "For a Parametric calculation of VaR, " \
                "both period & expected volatility are required"
            )

        #* Formula for Parametric Method:
        # VaR = Investment Value * Z-score * Expected Volatility * sqrt (Time Horizon / trading days)
        z_score = NormalDist().inv_cdf(confidence_level / 100)
        
        VaR = abs(investment_value * z_score * expected_volatility * sqrt(period / BUSINESS_DAYS_IN_YEAR))

        return VaR
    
    elif (method == VaRMethod.HISTORICAL):
        if (historical_returns is None or type(historical_returns) != list):
            raise TypeError(
                "For a Historical Simulation calculation of VaR, " \
                "a list of historical return percentages is required"
            )
        
        nth_percentile_percent = np.percentile(historical_returns, 100-confidence_level)

        VaR = abs(nth_percentile_percent * investment_value)

        return VaR


def main():
    VaR = get_value_at_risk(VaRMethod.PARAMETRIC, investment_value=100_000, confidence_level=99, expected_volatility=0.2, period=10)
    print(VaR)
    VaR = get_value_at_risk(VaRMethod.HISTORICAL, investment_value=100_000, confidence_level=99, historical_returns=[0.009, 0.01, 0.05, -0.005, -0.030])
    print(VaR)

if __name__ == "__main__":
    main()
    