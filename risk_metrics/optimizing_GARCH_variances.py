from risk_handler import RiskHandler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def observed_variances(daily_returns : pd.DataFrame):
    squared_returns = daily_returns**2
    squared_returns.dropna(inplace=True, how='all')

    observations = squared_returns[::-1].rolling(window=25).mean().dropna(how='all')
    
    return observations[::-1]

def MAPE(observations, variances, instrument):
    MAPE = (abs(observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance']) / observations[f'{instrument}_return']).mean()
    return MAPE

def RMSE(observations, variances, instrument):
    RMSE = ((observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance'])**2).mean()**0.5
    return RMSE

def MAE(observations, variances, instrument):
    MAE = abs(observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance']).mean()
    return MAE

def inverse_R_squared(observations, variances, instrument):
    SS_res = ((observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance'])**2).sum()
    SS_tot = ((observations[f'{instrument}_return'] - observations[f'{instrument}_return'].mean())**2).sum()
    inverse_R_squared = (SS_res / SS_tot)
    return inverse_R_squared

def MASE(observations, variances, instrument):
    MAE = abs(observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance']).mean()
    naive_MAE = abs(observations[f'{instrument}_return'] - observations[f'{instrument}_return'].shift(1)).mean()
    MASE = MAE / naive_MAE
    return MASE

def theil_U(observations, variances, instrument):
    SS_res = ((observations[f'{instrument}_return'] - variances[f'{instrument}_garch_variance'])**2).sum()
    SS_naive = ((observations[f'{instrument}_return'] - observations[f'{instrument}_return'].shift(1))**2).sum()
    SS_naive2 = ((observations[f'{instrument}_return'].shift(1) - observations[f'{instrument}_return'].shift(2))**2).sum()
    theil_U = (SS_res / SS_naive) / (SS_naive / SS_naive2)
    return theil_U

def standardized_RMSE(observations : pd.DataFrame, variances, instrument):
    standardized_RMSE = RMSE(observations, variances, instrument) / (observations[f'{instrument}_return'].max()-observations[f'{instrument}_return'].min())
    return standardized_RMSE

def full_funct(weights, funct, rh : RiskHandler, observations_or_returns):
    rh.set_omega_based_GARCH_variance_weights(weights)
    
    funct_result = funct(observations_or_returns, rh.Omega_based_GARCH_variances, 'ES_data')

    return funct_result

def maximum_likelihood(returns, variances, instrument):
    total = 0
    returns.dropna(inplace=True, how='all')
    variances.dropna(inplace=True, how='all')
    
    dates = returns.index.intersection(variances.index)

    for i in range(1, len(dates)):
        if np.isnan(variances[f'{instrument}_garch_variance'].loc[dates[i]]):
            continue
        if variances[f'{instrument}_garch_variance'].loc[dates[i]] == 0:
            continue
        variance_estimate = variances[f'{instrument}_garch_variance'].loc[dates[i]]
        return_squared = returns[f'{instrument}_return'].loc[dates[i]]**2
        value = -np.log(variance_estimate) - return_squared / variance_estimate
        total += value
    # needs to be negative since we are minimizing (but want to maximize the likelihood)
    return -total

def constraint_function(weights):
    # Ensure the sum of weights is equal to 1
    return sum(weights) - 1

def drop_first_N_rows(df, N):
    new_df = df.iloc[N:]
    return new_df

if __name__ == '__main__':
    trend_tables = {}
    
    instruments = ['ES_data']#, 'ZN_data', 'RB_data']

    for instrument in instruments:
        trend_tables[instrument] = pd.read_csv(f'unittesting/test_data/{instrument}.csv')

    

    rh = RiskHandler(trend_tables=trend_tables)
    daily_returns = rh.daily_returns
    print(daily_returns.to_csv('returns.csv'))
    # initialize this
    rh.long_term_variances

    observations = observed_variances(daily_returns)

    # cook_time = 2560

    # daily_returns = drop_first_N_rows(daily_returns, cook_time)
    # observations = drop_first_N_rows(observations, cook_time)

    initial_guess = [0.01, 0.69, 0.30]

    bounds = [(0, 1), (0, 1), (0, 1)]

    constraints = ({'type': 'eq', 'fun': constraint_function})

    args = (maximum_likelihood, rh, daily_returns)

    result = minimize(
        full_funct,
        initial_guess, 
        bounds=bounds, 
        constraints=constraints, 
        args=args)


    print(result)
    print(result.x)

    rh.set_GARCH_variance_weights(tuple(result.x))
    GARCH_variances = rh.GARCH_variances.dropna(how='all')
    

    # Graph each of the GARCH_variances
    # for instrument in instruments:
    plt.plot(GARCH_variances.index, GARCH_variances[f'ES_data_garch_variance'], label=f'ES_data_garch_variance')

    plt.xlabel('Date')
    plt.ylabel('GARCH Variance')
    plt.legend()
    plt.show()