import numpy as np

def max_leverage_position_limit(maximum_leverage : float, capital : float, notional_exposure_per_contract : float | np.ndarray) -> float | np.ndarray:
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
    """
    return maximum_leverage * capital / notional_exposure_per_contract

def max_forecast_position_limit(
        maximum_forecast_ratio : float, 
        capital : float, 
        IDM : float, 
        tau : float,
        max_forecast_buffer : float,
        instrument_weight : float | np.ndarray,
        notional_exposure_per_contract : float | np.ndarray, 
        annualized_volatility : float | np.ndarray) -> float | np.ndarray:
    
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
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    return (1 + max_forecast_buffer) * maximum_forecast_ratio * capital * IDM * instrument_weight * tau / notional_exposure_per_contract / annualized_volatility

def max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest : float, open_interest : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the lesser of the max acceptable percentage of open interest and the number of contracts to be traded
    (works for both single instruments and arrays)

    Parameters:
    ---
        max_acceptable_pct_of_open_interest : float
            the max acceptable percentage of open interest for a given instrument
        open_interest : float | np.ndarray
            the open interest for the instrument
    """
    return max_acceptable_pct_of_open_interest * open_interest

def position_limit_aggregator(
    maximum_position_leverage : float,
    capital : float,
    IDM : float,
    tau : float,
    maximum_forecast_ratio : float,
    max_acceptable_pct_of_open_interest : float,
    max_forecast_buffer : float,
    contracts : float | np.ndarray,
    notional_exposure_per_contract : float | np.ndarray,
    annualized_volatility : float | np.ndarray,
    instrument_weight : float | np.ndarray,
    open_interest : float | np.ndarray) -> float | np.ndarray:
    """
    Returns the minimum of the three position limits
    (works for both single instruments and arrays)

    Parameters:
    ---
        maximum_leverage : float
            the max acceptable leverage for a given instrument
        capital : float
            the total capital allocated to the portfolio
        IDM : float
            instrument diversification multiplier
        tau : float
            the target risk for the portfolio
        maximum_forecast_ratio : float
            the max forecast ratio (max forecast / average forecast) ... often 20 / 10 -> 2
        max_acceptable_pct_of_open_interest : float
            the max acceptable percentage of open interest for a given instrument
        max_forecast_buffer : float
            the max acceptable buffer for the forecast
        contracts : float | np.ndarray
            the number of contracts to be traded
        notional_exposure_per_contract : float | np.ndarray
            the notional exposure per contract for the instrument
        annualized_volatility : float | np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
        instrument_weight : float | np.ndarray
            the weight of the instrument in the portfolio (capital allocated to the instrument / total capital)
            ... often 1/N
        open_interest : float | np.ndarray
            the open interest for the instrument
    """
    if isinstance(contracts, (int, float)):
        return min(
            max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract),
            max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility),
            max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest, open_interest), contracts)
    
    return np.minimum(np.minimum(
        max_leverage_position_limit(maximum_position_leverage, capital, notional_exposure_per_contract),
        max_forecast_position_limit(maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight, notional_exposure_per_contract, annualized_volatility),
        max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest, open_interest)), contracts)

