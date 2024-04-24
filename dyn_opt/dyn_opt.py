import pandas as pd
import numpy as np
from functools import reduce
from risk_limits import portfolio_risk, position_risk

def get_notional_exposure_per_contract(unadj_prices : pd.DataFrame, multipliers : pd.DataFrame) -> pd.DataFrame:
    notional_exposure_per_contract = unadj_prices.apply(lambda col: col * multipliers.loc['Multiplier', col.name])
    return notional_exposure_per_contract.abs()

def get_weight_per_contract(notional_exposure_per_contract : pd.DataFrame, capital : float) -> pd.DataFrame:
    return notional_exposure_per_contract / capital

def get_cost_penalty(x : np.ndarray, y : np.ndarray, weighted_cost_per_contract : np.ndarray, cost_penalty_scalar : int) -> float:
    """Finds the trading cost to go from x to y, given the weighted cost per contract and the cost penalty scalar"""

    #* Should never activate but just in case
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    weighted_cost_per_contract = np.nan_to_num(np.asarray(weighted_cost_per_contract, dtype=np.float64))

    trading_cost = np.abs(x - y) * weighted_cost_per_contract

    return np.sum(trading_cost) * cost_penalty_scalar

def get_portfolio_tracking_error_standard_deviation(x : np.ndarray, y : np.ndarray, covariance_matrix : np.ndarray, cost_penalty : float = 0.0) -> float:
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(covariance_matrix).any():
        raise ValueError("Input contains NaN values")
    
    tracking_errors = x - y

    radicand = tracking_errors @ covariance_matrix @ tracking_errors.T

    #* deal with negative radicand (really, REALLY shouldn't happen)
    #? maybe its a good weight set but for now, it's probably safer this way
    if radicand < 0:
        return 1.0 # Return 100% TE
    
    return np.sqrt(radicand) + cost_penalty

def covariance_row_to_matrix(row : np.ndarray) -> np.ndarray:
    num_instruments = int(np.sqrt(2 * len(row)))
    matrix = np.zeros((num_instruments, num_instruments))

    idx = 0
    for i in range(num_instruments):
        for j in range(i, num_instruments):
            matrix[i, j] = matrix[j, i] = row[idx]
            idx += 1

    return matrix

def round_multiple(x : np.ndarray, multiple : np.ndarray) -> np.ndarray:
    return np.round(x / multiple) * multiple

def buffer_weights(optimized : np.ndarray, held : np.ndarray, weights : np.ndarray, covariance_matrix : np.ndarray, tau : float, asymmetric_risk_buffer : float):
    tracking_error = get_portfolio_tracking_error_standard_deviation(optimized, held, covariance_matrix)

    tracking_error_buffer = tau * asymmetric_risk_buffer

    # If the tracking error is less than the buffer, we don't need to do anything
    if tracking_error < tracking_error_buffer:
        return held
    
    adjustment_factor = max((tracking_error - tracking_error_buffer) / tracking_error, 0.0)

    required_trades = (optimized - held) * adjustment_factor

    return round_multiple(held + required_trades, weights)

# Might be worth framing this similar to scipy.minimize function in terms of argument names (or quite frankly, maybe just use scipy.minimize)
def greedy_algorithm(ideal : np.ndarray, x0 : np.ndarray, weighted_costs_per_contract : np.ndarray, held : np.ndarray, weights_per_contract : np.ndarray, covariance_matrix : np.ndarray, cost_penalty_scalar : int) -> np.ndarray:
    if ideal.ndim != 1 or ideal.shape != x0.shape != held.shape != weights_per_contract.shape != weighted_costs_per_contract.shape:
        raise ValueError("Input shapes do not match")
    if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1] or len(ideal) != covariance_matrix.shape[0]:
        raise ValueError("Invalid covariance matrix (should be [N x N])")
    
    proposed_solution = x0.copy()
    cost_penalty = get_cost_penalty(held, proposed_solution, weighted_costs_per_contract, cost_penalty_scalar)
    tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, proposed_solution, covariance_matrix, cost_penalty)
    best_tracking_error = tracking_error
    iteration_limit = 1000
    iteration = 0

    while iteration <= iteration_limit:
        previous_solution = proposed_solution.copy()
        best_IDX = None

        for idx in range(len(proposed_solution)):
            temp_solution = previous_solution.copy()

            if temp_solution[idx] < ideal[idx]:
                temp_solution[idx] += weights_per_contract[idx]
            else:
                temp_solution[idx] -= weights_per_contract[idx]

            cost_penalty = get_cost_penalty(held, temp_solution, weighted_costs_per_contract, cost_penalty_scalar)
            tracking_error = get_portfolio_tracking_error_standard_deviation(ideal, temp_solution, covariance_matrix, cost_penalty)

            if tracking_error <= best_tracking_error:
                best_tracking_error = tracking_error
                best_IDX = idx

        if best_IDX is None:
            break

        if proposed_solution[best_IDX] <= ideal[best_IDX]:
            proposed_solution[best_IDX] += weights_per_contract[best_IDX]
        else:
            proposed_solution[best_IDX] -= weights_per_contract[best_IDX]
        
        iteration += 1

    return proposed_solution

def clean_data(*args):
    dfs = [df.set_index(pd.to_datetime(df.index)).dropna() for df in args]
    intersection_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))
    dfs = [df.loc[intersection_index].astype(np.float64) for df in dfs]

    return dfs

def iterator(
        covariances : pd.DataFrame,
        jump_covariances : pd.DataFrame,
        ideal_positions_weighted : pd.DataFrame, 
        weight_per_contract : pd.DataFrame, 
        costs_per_contract_weighted : pd.DataFrame,
        notional_exposure_per_contract : pd.DataFrame,
        open_interest : pd.DataFrame,
        instrument_weight : np.ndarray,
        tau : float,
        capital : float,
        IDM : float,
        maximum_forecast_ratio : float,
        maximum_position_leverage : float,
        max_acceptable_pct_of_open_interest : float,
        max_forecast_buffer : float,
        maximum_portfolio_leverage : float,
        maximum_correlation_risk : float, 
        maximum_portfolio_risk : float,
        maximum_jump_risk : float,
        asymmetric_risk_buffer : float) -> pd.DataFrame:
    #@ Data cleaning
    ideal_positions_weighted, weight_per_contract, costs_per_contract_weighted, covariances = clean_data(ideal_positions_weighted, weight_per_contract, costs_per_contract_weighted, covariances)

    # Make sure they all have the same columns, and order !!
    intersection_columns = ideal_positions_weighted.columns.intersection(weight_per_contract.columns).intersection(costs_per_contract_weighted.columns)
    ideal_positions_weighted = ideal_positions_weighted[intersection_columns]
    weight_per_contract = weight_per_contract[intersection_columns]
    costs_per_contract_weighted = costs_per_contract_weighted[intersection_columns]

    dates = ideal_positions_weighted.index

    # Initialize x0 and cost penalty scalar
    x0 = np.zeros(len(ideal_positions_weighted.columns))
    cost_penalty_scalar = 10

    optimized_positions = pd.DataFrame(index=dates, columns=ideal_positions_weighted.columns)
    optimized_positions = optimized_positions.astype(np.float64)

    for n, date in enumerate(dates):
        ideal_positions_weighted_one_day = ideal_positions_weighted.loc[date].values
        costs_per_contract_weighted_one_day = costs_per_contract_weighted.loc[date].values
        covariance_matrix_one_day = covariance_row_to_matrix(covariances.loc[date].values)
        weight_per_contract_one_day = weight_per_contract.loc[date].values
        notional_exposure_per_contract_one_day = notional_exposure_per_contract.loc[date].values
        open_interest_one_day = open_interest.loc[date].values
        jump_covariance_matrix_one_day = covariance_row_to_matrix(jump_covariances.loc[date].values)
        instrument_weight_one_day = instrument_weight.loc[date].values

        held_positions_weighted = np.zeros(len(ideal_positions_weighted.columns))

        if n != 0:
            current_date_IDX = dates.get_loc(date)
            held_positions_weighted = optimized_positions.iloc[current_date_IDX - 1].values * weight_per_contract_one_day

        optimized_weights_one_day = greedy_algorithm(ideal_positions_weighted_one_day, x0, costs_per_contract_weighted_one_day, held_positions_weighted, weight_per_contract_one_day, covariance_matrix_one_day, cost_penalty_scalar)

        buffered_weights = buffer_weights(
            optimized_weights_one_day, held_positions_weighted, weight_per_contract_one_day, 
            covariance_matrix_one_day, tau, asymmetric_risk_buffer)

        optimized_positions_one_day = buffered_weights / weight_per_contract_one_day

        STD = np.sqrt(np.diag(covariance_matrix_one_day))

        risk_limited_positions = position_risk.position_limit_aggregator(
            maximum_position_leverage, capital, IDM, tau, maximum_forecast_ratio, 
            max_acceptable_pct_of_open_interest, max_forecast_buffer, optimized_positions_one_day, 
            notional_exposure_per_contract_one_day, STD, instrument_weight_one_day, open_interest_one_day)

        risk_limited_positions_weighted = risk_limited_positions * weight_per_contract_one_day

        portfolio_risk_limited_positions = portfolio_risk.portfolio_risk_aggregator(
            risk_limited_positions, risk_limited_positions_weighted, covariance_matrix_one_day, 
            jump_covariance_matrix_one_day, maximum_portfolio_leverage, maximum_correlation_risk, 
            maximum_portfolio_risk, maximum_jump_risk)

        optimized_positions.loc[date] = portfolio_risk_limited_positions

    return optimized_positions

def aggregator(
    capital : float,
    fixed_cost_per_contract : float,
    tau : float,
    asymmetric_risk_buffer : float,
    unadj_prices : pd.DataFrame,
    multipliers : pd.DataFrame,
    ideal_positions : pd.DataFrame,
    covariances : pd.DataFrame,
    jump_covariances : pd.DataFrame,
    open_interest : pd.DataFrame,
    instrument_weight : pd.DataFrame,
    IDM : float,
    maximum_forecast_ratio : float,
    max_acceptable_pct_of_open_interest : float,
    max_forecast_buffer : float,
    maximum_position_leverage : float,
    maximum_portfolio_leverage : float,
    maximum_correlation_risk : float,
    maximum_portfolio_risk : float,
    maximum_jump_risk : float)-> pd.DataFrame:

    unadj_prices, ideal_positions, covariances, jump_covariances, open_interest, instrument_weight = clean_data(unadj_prices, ideal_positions, covariances, jump_covariances, open_interest, instrument_weight)

    multipliers = multipliers.sort_index(axis=1)

    notional_exposure_per_contract = get_notional_exposure_per_contract(unadj_prices, multipliers)
    weight_per_contract = get_weight_per_contract(notional_exposure_per_contract, capital)

    ideal_positions_weighted = ideal_positions * weight_per_contract

    costs_per_contract = pd.DataFrame(index=ideal_positions_weighted.index, columns=ideal_positions_weighted.columns).fillna(fixed_cost_per_contract)
    costs_per_contract_weighted = costs_per_contract / capital / weight_per_contract

    return iterator(
        covariances, jump_covariances, ideal_positions_weighted, weight_per_contract,
        costs_per_contract_weighted, notional_exposure_per_contract, open_interest, 
        instrument_weight, tau, capital, IDM, maximum_forecast_ratio, maximum_position_leverage, 
        max_acceptable_pct_of_open_interest, max_forecast_buffer, maximum_portfolio_leverage, 
        maximum_correlation_risk, maximum_portfolio_risk, maximum_jump_risk, asymmetric_risk_buffer)
