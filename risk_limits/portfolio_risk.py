import numpy as np

def max_leverage_portfolio_multiplier(maximum_portfolio_leverage : float, positions_weighted : np.ndarray) -> float:
    """
    Returns the positions scaled by the max leverage limit

    Parameters:
    ---
        maximum_portfolio_leverage : float
            the max acceptable leverage for the portfolio
        positions_weighted : np.ndarray
            the notional exposure / position * # positions / capital
            Same as dynamic optimization
    """
    leverage = np.sum(np.abs(positions_weighted))
    scalar = np.minimum(maximum_portfolio_leverage / leverage, 1)

    return scalar

def correlation_risk_portfolio_multiplier(maximum_portfolio_correlation_risk : float, positions_weighted : np.ndarray, STD : np.ndarray) -> float:
    """
    Returns the positions scaled by the correlation risk limit

    Parameters:
    ---
        positions_weighted : np.ndarray
            the notional exposure / position * # positions / capital
            Same as dynamic optimization
        STD : np.ndarray
            standard deviation of returns for the instrument, in same terms as tau e.g. annualized
    """
    correlation_risk = np.sum(np.abs(positions_weighted) * STD)
    scalar = np.minimum(1, maximum_portfolio_correlation_risk / correlation_risk)

    return scalar

def portfolio_risk_multiplier(maximum_portfolio_volatility : float, positions_weighted : np.ndarray, covariance_matrix : np.ndarray) -> float:
    """
    Returns the positions scaled by the portfolio volatility limit

    Parameters:
    ---
        maximum_portfolio_volatility : float
            the max acceptable volatility for the portfolio
        positions_weighted : np.ndarray
            the notional exposure / position * # positions / capital
            Same as dynamic optimization
        covariance_matrix : np.ndarray
            the covariances between the instrument returns
    """
    portfolio_volatility = np.sqrt(positions_weighted @ covariance_matrix @ positions_weighted.T)
    scalar = np.minimum(1, maximum_portfolio_volatility / portfolio_volatility)

    return scalar

def jump_risk_multiplier(maximum_portfolio_jump_risk : float, positions_weighted : np.ndarray, jump_covariance_matrix) -> float:
    """
    Returns the positions scaled by the jump risk limit

    Parameters:
    ---
        maximum_portfolio_jump_risk : float
            the max acceptable jump risk for the portfolio
        positions_weighted : np.ndarray
            the notional exposure / position * # positions / capital
            Same as dynamic optimization
        jumps : np.ndarray
            the jumps in the instrument returns
    """
    jump_risk = np.sqrt(positions_weighted @ jump_covariance_matrix @ positions_weighted.T)
    scalar = np.minimum(1, maximum_portfolio_jump_risk / jump_risk)

    return scalar

def portfolio_risk_aggregator(
        positions : np.ndarray, 
        positions_weighted : np.ndarray, 
        covariance_matrix : np.ndarray, 
        jump_covariance_matrix : np.ndarray,
        maximum_portfolio_leverage : float,
        maximum_correlation_risk : float,
        maximum_portfolio_risk : float,
        maximum_jump_risk : float) -> np.ndarray:
    
    STD = np.sqrt(np.diag(covariance_matrix))

    leverage_multiplier = max_leverage_portfolio_multiplier(maximum_portfolio_leverage, positions_weighted)
    correlation_multiplier = correlation_risk_portfolio_multiplier(maximum_correlation_risk, positions_weighted, STD)
    volatility_multiplier = portfolio_risk_multiplier(maximum_portfolio_risk, positions_weighted, covariance_matrix)
    jump_multiplier = jump_risk_multiplier(maximum_jump_risk, positions_weighted, jump_covariance_matrix)

    return positions * np.minimum(leverage_multiplier, correlation_multiplier, volatility_multiplier, jump_multiplier)
