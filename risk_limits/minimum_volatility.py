import numpy as np

def minimum_volatility(max_forecast_ratio : float, IDM : float, tau : float, maximum_leverage : float, instrument_weight : float | np.ndarray, annualized_volatility : float | np.ndarray) -> bool:
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
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return annualized_volatility >= (max_forecast_ratio * IDM * instrument_weight * tau) / maximum_leverage
