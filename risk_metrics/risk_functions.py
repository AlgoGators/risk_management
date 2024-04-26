import numpy as np

def daily_variance_to_annualized_volatility(daily_variance : float | np.ndarray) -> float | np.ndarray:
    return (daily_variance * 256) ** 0.5