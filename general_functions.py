import pandas as pd
import numpy as np

def get_daily_returns(
    price_df : pd.DataFrame,
    date_column : str = "Date",
    price_column : str = "Close") -> pd.DataFrame:
    """
    Returns a dataframe of daily % returns with date as index
    
    Parameters:
    ---
        price_df : pd.DataFrame
            That includes a date column and price columns
        date_column : str (optional)
            The name of the date column
        price_column : str (optional)
            The name of the price column
    ---
    """

    # Skip the first date since we won't have a return on that day
    dates = price_df[date_column].tolist()[1:]

    returns = []

    prices = price_df[price_column].tolist()

    for i, price in enumerate(prices[1:]):
        returns.append((price - prices[i]) / prices[i])
        
    returns_df = pd.DataFrame.from_dict({"Date" : dates, "Returns" : returns})
    
    returns_df = returns_df.set_index("Date")
    
    return returns_df

def get_position_weight(
        number_of_contracts : int,
        notional_exposure_per_contract : float,
        capital : float) -> float:
    """
    returns the weight of the position's notional exposure to total portfolio capital

    parameters:
    ---
        number_of_contracts : int
            number of contracts allocated to this position with appropriate trend sign
            * negative if short
        notional_exposure_per_contract : float
            notional exposure per contract (price * multiplier * fx rate)
        capital : float
            total capital allocated to the portfolio
    ---
    """

    return number_of_contracts * notional_exposure_per_contract / capital