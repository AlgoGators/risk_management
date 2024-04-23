import pandas as pd
import numpy as np

def minimum_volatility(max_forecast_ratio : float, IDM : float, tau : float, maximum_leverage : float, instrument_weight : float | np.ndarray, STD : float | np.ndarray) -> float | np.ndarray:
    """
    Returns True if the returns for a given instrument meets a minimum level of volatility; else, False
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        STD : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return STD >= (max_forecast_ratio * IDM * instrument_weight * tau) / maximum_leverage

def max_leverage_position_limit(maximum_leverage : float, capital : float, notional_exposure_per_contract : float | np.ndarray, contracts : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the max leverage limit and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        capital : float
            the total capital allocated to the portfolio
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
        contracts : float | np.ndarray
            the number of contracts to be traded
    """
    return np.minimum(maximum_leverage * capital / notional_exposure_per_contract, contracts)

def max_forecast_position_limit(
        maximum_forecast_ratio : float, 
        capital : float, 
        IDM : float, 
        tau : float,
        instrument_weight : float | np.ndarray, 
        notional_exposure_per_contract : float | np.ndarray, 
        STD : float | np.ndarray,
        contracts : float | np.ndarray) -> float | np.ndarray:
    
    """
    Returns the lesser of the max forecast limit and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        capital : float
            the total capital allocated to the portfolio
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
        STD : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        contracts : float | np.ndarray
            the number of contracts to be traded
    """
    return np.minimum(maximum_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / STD, contracts)

def max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest : float, open_interest : float | np.ndarray, contracts : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the max acceptable percentage of open interest and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_acceptable_pct_of_open_interest : float
            the max acceptable percentage of open interest for a given instrument
        open_interest : float | np.ndarray
            the open interest for the instrument
        contracts : float | np.ndarray
            the number of contracts to be traded
    """
    return np.minimum(max_acceptable_pct_of_open_interest * open_interest, contracts)

