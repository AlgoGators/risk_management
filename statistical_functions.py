import numpy as np
import pandas as pd
from math import sqrt
from enum import Enum

# Ugly but it allows keeping the same import statement across submodules and parent directories
try:
    from .constants import BUSINESS_DAYS_IN_YEAR, BUSINESS_DAYS_IN_TEN_YEARS
except ImportError:
    from constants import BUSINESS_DAYS_IN_YEAR, BUSINESS_DAYS_IN_TEN_YEARS



class Periods(Enum):
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 20
    QUARTERLY = 60
    YEARLY = 256

def SMA( 
        lst : list[float], 
        span : int = -1
    ) -> float:

    if (span == -1):
        return np.nanmean(lst)
    
    return np.nanmean(lst[-span])


def SR(
        returns : list[float]) -> float:
    mean = np.nanmean(returns)

    return mean / std(returns)


def EWMA ( 
        lst : list[float], 
        span : int = None,
        alpha : float = None,
        threshold : int = 100) -> float:
    """Returns Exponentially Weighted Moving Average given span"""

    #Carver's fromula
    #@ EWMA(t) = α(1 - α)⁰ * y(t) + α(1 - α)¹ * y(t-1) + α(1-α)² * y(t-2) + ... + α(1-α)ⁿ * y(t-n)
    # where alpha is 2 / (span + 1) & n is the length of the list

    # I prefer to use the standard EWMA instead of Carver's 
    # as a it results in the mean being the same as the SMA when all the values are equal
    #@ EWMA(t) = α * y(t) + (1 - α) * EWMA(t-1)

    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")
    

    if alpha is None:
        alpha : float = 2 / (span + 1)


    lst_len : int = len(lst)
    last_IDX : int = lst_len - 1

    if (lst_len <= threshold):
        return SMA(lst)
    
    ewma : float = SMA(lst[:threshold])


    for n in range(threshold, lst_len):
        ewma = alpha * lst[n] + (1 - alpha) * ewma
        # ewma += (alpha * (1-alpha)**n * lst[last_IDX - n])

    return ewma


def std(
        lst : list[float],
        span : int = -1,
        annualize : bool = False) -> float:
    
    """Returns the standard deviation of a list of values"""

    # get the lesser of the two values
    span = min(len(lst), span)

    reduced_lst : list[float] = lst[:-span] if (span != -1) else lst

    xbar : float = np.mean(reduced_lst)

    numerator : float = 0

    for x in reduced_lst:
        numerator += (x - xbar)**2

    lst_len : int = len(reduced_lst)

    if lst_len == 1:
        return 0

    standard_deviation : float = sqrt(numerator / (len(reduced_lst) - 1))

    factor = 1

    if (annualize is True):
        factor = sqrt(BUSINESS_DAYS_IN_YEAR)

    return standard_deviation * factor 


def VAR(
    lst : list[float],
    span : int = -1,
    annualize : bool = False) -> float:

    return std(lst=lst, span=span, annualize=annualize)**2


def exponentially_weighted_stddev(
        lst : list,
        span : int = None, 
        alpha : float = None, 
        annualize : bool = False,
        threshold : int = 100) -> float:
    """
    """

    #@ given an expoentially weighted moving average, r*
    #@                     _________________________________________________________________________________
    #@ exponential σ(t) = √ α(1 - α)⁰(r(t) - r*)² + α(1 - α)¹(r(t-1) - r*)² + α(1 - α)²(r(t-2) - r*)² + ... 


    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")

    ewma : float = EWMA(lst, span=span, alpha=alpha, threshold=threshold)
    
    if alpha is None:
        alpha : float = 2 / (span + 1)

    radicand : float  = 0
    lst_len : int = len(lst)
    last_IDX : int = lst_len - 1

    if (lst_len <= threshold):
        return std(lst=lst, span=span, annualize=annualize)

    # starting value is just the simple variance of the first 100 values (threshold). Variance is the radicand of the stddev formula
    radicand = VAR(lst[:threshold])

    for n in range(threshold, lst_len):
        radicand += (alpha * (1 - alpha)**n * (lst[last_IDX - n] - ewma)**2) 

    ew_stddev : float = sqrt(radicand)

    factor = 1
    if (annualize is True):
        factor = sqrt(BUSINESS_DAYS_IN_YEAR)

    return ew_stddev * factor


def exponentially_weighted_var(
    lst : list[float],
    alpha : float = None,
    span : int = None,
    threshold : int = 10) -> float:

    """
        threshold : when the formula switches from a simple variance to EW_var
    """

    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")
    
    if alpha is None:
        alpha : float = 2 / (span + 1)

    lst_len : int = len(lst)


    # if the list length is less than threshold just return the simple variance (stddev squared)
    if lst_len <= threshold:
        return VAR(lst=lst)
    

    # EWV(t) = λ * (x(t) − EWMA(t−1))^2 + (1 − λ) * EWV(t−1)
    # starting value is just the simple variance of the first 10 values
    # EW_var = stddev(lst[:10])**2

    # starting with n = 10 (threshold)
    # EW_var = alpha * (lst[n] - EWMA(lst[:n], alpha=alpha))**2 + (1-alpha) * EW_var
    
    EW_var : float = VAR(lst[:threshold])

    for n in range(threshold, lst_len):
        EW_var = alpha * (lst[n] - EWMA(lst[:n], alpha=alpha))**2 + (1 - alpha) * EW_var

    return EW_var


def exponentially_weighted_covar(
    lst_X : list[float],
    lst_Y : list[float],
    alpha : float = None,
    span : int = None) -> float:
    """Calculates an exponentially weighted covariance with weight alpha"""

    #@ EW Cov(X, Y) = sum(0, t) (1 - α)^n * (X(t-1) - EWMA(X)) * (Y(t-1) - EWMA(Y))

    # checks that only one of the variables is given
    if not(any([span, alpha]) and not all([span, alpha])):
        raise ValueError("Only one of span or alpha may be used")
    
    if alpha is None:
        alpha : float = 2 / (span + 1)

    EW_covar : float = 0

    lst_len : int = len(lst_X)
    last_IDX : int = lst_len - 1

    ewma_X : float = EWMA(lst_X, alpha=alpha)
    ewma_Y : float = EWMA(lst_Y, alpha=alpha)

    for n in range(0, lst_len):
        EW_covar += (1-alpha)**n * (lst_X[last_IDX-n] - ewma_X) * (lst_Y[last_IDX-n] - ewma_Y)

    return EW_covar


def correlation(
    returns_X : pd.DataFrame,
    returns_Y : pd.DataFrame) -> float:
    """Calculates a correlation (rho) between two DataFrames where each dataframe had a "Date" column"""

    rho = 0.0

    # Try to merge the two dataframes on the date column
    try:
        merged_df = pd.merge(returns_X, returns_Y, on="Date", how="inner")
        rho = merged_df.iloc[:,1].corr(merged_df.iloc[:,2])
        
    # If not just merge them on the index
    except KeyError:
        merged_df = pd.merge(returns_X, returns_Y, left_index=True, right_index=True, how="inner")
        rho = merged_df.iloc[:,0].corr(merged_df.iloc[:,1])

    return rho


def exponentially_weighted_correlation(
    returns_X : pd.DataFrame,
    returns_Y : pd.DataFrame,
    span : int) -> float:
    
    covar : float = exponentially_weighted_covar(returns_X, returns_Y, span=span)

    var_tickerX : float = exponentially_weighted_var(returns_X, span=span)
    var_tickerY : float = exponentially_weighted_var(returns_Y, span=span)


    # default value for correlation should be nan just in case variance is 0
    correlation : float = np.nan 
    
    # make sure neither variance is 0, if so then calculate the correlation
    if (0 not in [var_tickerX, var_tickerY]):
        correlation = covar / (sqrt(var_tickerX * var_tickerY))

    return correlation


def correlation_matrix(
    returns_df : pd.DataFrame,
    period : Periods,
    window : int) -> np.array:

    periodic_returns_df = pd.DataFrame()
    
    tickers = returns_df.columns.tolist()

    for ticker in tickers:
        returns = returns_df[ticker].tolist()

        # groups them and takes the recent window backwards
        periodic_returns_df[ticker] = [sum(returns[x : x + period.value])
                                for x in range(0, len(returns), period.value)][:-window]

    correlation_matrix : list[float] = []

    for n, tickerX in enumerate(tickers):
        # always start with 1.0 in the correlations since X1 is perfectly correlated with X2
        correlation_lst : list[float] = [1.0]

        # go through the remaining variables, ignoring those previously calculated
        for tickerY in tickers[n+1:]:
            rho = correlation(periodic_returns_df[tickerX], periodic_returns_df[tickerY])

            correlation_lst.append(rho)

        correlation_matrix.append(correlation_lst)

    #? right now we these correlations
    #? coordinates are relative to the existing values
    #?     0   1   2   3   4
    #?     A   B   C   D   E
    #? 0 A 1.0 0,1 0,2 0,3   
    #? 1 B 0.5 1.0 1,1 1,2
    #? 2 C 0.6 0.7 1.0 2,1
    #? 3 D 0.4 0.9 0.4 1.0
    #? 4 E 0.8 0.7 0.9 0.4 1.0

    correlation_df = pd.DataFrame(columns=tickers)

    for n, column in enumerate(correlation_matrix):
        # we need the second value in each column to the left 

        current_ticker : str = tickers[n]

        if (n == 0):
            correlation_df[current_ticker] = column
            continue

        new_column = []

        row : int = n

        for x in range(0,n):
            new_column.append(correlation_matrix[x][row])
            row -= 1

        new_column.extend(column)

        correlation_df[current_ticker] = new_column

    return np.array(correlation_df)


def exponentially_weighted_correlation_matrix(
    returns_df : pd.DataFrame,
    period : Periods,
    span : int) -> np.array:

    #@ correlation formula
    #@ r =             EW Cov(X,Y)
    #@     -------------------------------
    #@      ______________________________
    #@     √    EW Var(X) * EW Var(Y)

    periodic_returns_df = pd.DataFrame()
    
    tickers = returns_df.columns.tolist()

    for ticker in tickers:
        returns = returns_df[ticker].tolist()

        periodic_returns_df[ticker] = [sum(returns[x : x + period.value])
                                for x in range(0, len(returns), period.value)]

    correlation_matrix : list[float] = []

    for n, tickerX in enumerate(tickers):
        # always start with 1.0 in the correlations since X1 is perfectly correlated with X2
        correlations : list[float] = [1.0]

        # go through the remaining variables, ignoring those previously calculated
        for tickerY in tickers[n+1:]:
            correlation = exponentially_weighted_correlation(periodic_returns_df[tickerX], periodic_returns_df[tickerY], span=span)

            correlations.append(correlation)

        correlation_matrix.append(correlations)

    #? right now we these correlations
    #? coordinates are relative to the existing values
    #?     0   1   2   3   4
    #?     A   B   C   D   E
    #? 0 A 1.0 0,1 0,2 0,3   
    #? 1 B 0.5 1.0 1,1 1,2
    #? 2 C 0.6 0.7 1.0 2,1
    #? 3 D 0.4 0.9 0.4 1.0
    #? 4 E 0.8 0.7 0.9 0.4 1.0

    correlation_df = pd.DataFrame(columns=tickers)

    for n, column in enumerate(correlation_matrix):
        # we need the second value in each column to the left 

        current_ticker : str = tickers[n]

        if (n == 0):
            correlation_df[current_ticker] = column
            continue

        new_column = []

        row : int = n

        for x in range(0,n):
            new_column.append(correlation_matrix[x][row])
            row -= 1

        new_column.extend(column)

        correlation_df[current_ticker] = new_column

    return np.array(correlation_df)


def rolling_std(
    returns : pd.DataFrame,
    ten_year_weight : float = 0.3) -> float:
    """Calculates a rolling standard deviation for a given dataframe with weighting on the annualized stddev and 10 year average"""

    annualized_stddevs = []
    ten_year_averages = []

    # max values included in ew_stddev, this should expedite the process
    maximum_values = 100
    
    weighted_stddevs = []

    for n, val in enumerate(returns.tolist()):
        start = max(0, n - maximum_values)
        annualized_stddev = exponentially_weighted_stddev(returns[start:n+1], span=32, annualize=True)

        annualized_stddevs.append(annualized_stddev)

        if n < BUSINESS_DAYS_IN_TEN_YEARS:
            ten_year_average = np.mean(annualized_stddevs[:n+1])
            
        else:
            ten_year_average =np.mean(annualized_stddevs[n-BUSINESS_DAYS_IN_TEN_YEARS:n+1])

        ten_year_averages.append(ten_year_average)

        weighted_stddev = ten_year_weight * ten_year_average + (1 - ten_year_weight) * annualized_stddev

        weighted_stddevs.append(weighted_stddev)

    return weighted_stddevs


def portfolio_covar( 
        position_percent_returns : pd.DataFrame) -> np.array:
    """Calculates a covariance matrix as outlined by Carver on pages 606-607"""

    #@ Σ = σ.ρ.σᵀ = σσᵀ ⊙ ρ (using Hadamard product) = Diag(σ) * ρ * Diag(σ)
    #@ where:
    #@ ρ is the correlation matrix
    #@ σ is the vector of annualized estimates of % standard deviations 
    #@ use 32 day span for standard deviations
    #@ window for equally weighted correlation matrix of 52 weeks


    stddev_lst = []

    tickers = position_percent_returns.columns.tolist()

    for ticker in tickers:
        # get the most recent value
        rolling_stddev = rolling_std(position_percent_returns[ticker])[-1]
        stddev_lst.append(rolling_stddev)

    stddev_array = np.array(stddev_lst)

    corr_matrix = correlation_matrix(position_percent_returns, Periods.WEEKLY, 52)

    EW_covar = np.dot(np.dot(np.diag(stddev_array), corr_matrix), np.diag(stddev_array))

    return EW_covar


def portfolio_stddev(
        position_weights : pd.DataFrame,
        position_percent_returns : pd.DataFrame) -> float:
    
    #@                _______
    #@ Portfolio σ = √ w Σ wᵀ
    #@ w is the vector of positions weights, and Σ is the covariance matrix of percent returns 

    tickers : list = position_weights.columns.tolist()

    weights_lst : list = []

    # gets the weights for each instrument
    for ticker in tickers:
        weights_lst.append(position_weights.iloc[0, position_weights.columns.get_loc(ticker)])

    weights = np.array(weights_lst)

    weights_T = weights.transpose()

    covariance_matrix = portfolio_covar(position_percent_returns)

    radicand : float = np.dot(np.dot(weights, covariance_matrix), weights_T)

    return sqrt(radicand)
