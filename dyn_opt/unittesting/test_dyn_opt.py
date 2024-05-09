import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd
import numpy as np

from dyn_opt import dyn_opt

class TestDynamicOptimization(unittest.TestCase):
    def test_dyn_opt(self):
        capital = 500_000
        tau = 0.20
        asymmetric_risk_buffer = 0.05
        unadj_prices = pd.read_parquet('dyn_opt/unittesting/data/unadj_prices.parquet')
        multipliers = pd.read_parquet('dyn_opt/unittesting/data/multipliers.parquet')
        ideal_positions = pd.read_parquet('dyn_opt/unittesting/data/ideal_positions.parquet')
        covariances = pd.read_parquet('dyn_opt/unittesting/data/covariances.parquet')
        jump_covariances = pd.read_parquet('dyn_opt/unittesting/data/jump_covariances.parquet')
        open_interest = pd.read_parquet('dyn_opt/unittesting/data/open_interest.parquet')

        fixed_cost_per_contract = 3.0
        instrument_weight = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)
        IDM = 2.50
        maximum_forecast_ratio = 2.0
        maximum_portfolio_risk = 0.30
        maximum_jump_risk = 0.75
        maximum_position_leverage = 4.0
        maximum_correlation_risk = 0.65
        maximum_portfolio_leverage = 20.0
        max_acceptable_pct_of_open_interest = 0.01
        max_forecast_buffer = 0.50
        cost_penalty_scalar = 10

        # only use last 500 positions (everything else should take care of itself in the code)
        ideal_positions = ideal_positions[-500:]

        df = dyn_opt.aggregator(
            capital,
            fixed_cost_per_contract,
            tau,
            asymmetric_risk_buffer,
            unadj_prices,
            multipliers,
            ideal_positions,
            covariances,
            jump_covariances,
            open_interest,
            instrument_weight,
            IDM,
            maximum_forecast_ratio,
            max_acceptable_pct_of_open_interest,
            max_forecast_buffer,
            maximum_position_leverage,
            maximum_portfolio_leverage,
            maximum_correlation_risk,
            maximum_portfolio_risk,
            maximum_jump_risk,
            cost_penalty_scalar)

        # Only optimized for last 500 values
        expected_df = pd.read_parquet('dyn_opt/unittesting/data/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        pd.testing.assert_frame_equal(df, expected_df)

    def test_single_day(self):
        capital = 500_000
        tau = 0.20
        asymmetric_risk_buffer = 0.05
        unadj_prices = pd.read_parquet('dyn_opt/unittesting/data/unadj_prices.parquet')
        multipliers = pd.read_parquet('dyn_opt/unittesting/data/multipliers.parquet')
        ideal_positions = pd.read_parquet('dyn_opt/unittesting/data/ideal_positions.parquet')
        covariances = pd.read_parquet('dyn_opt/unittesting/data/covariances.parquet')
        jump_covariances = pd.read_parquet('dyn_opt/unittesting/data/jump_covariances.parquet')
        open_interest = pd.read_parquet('dyn_opt/unittesting/data/open_interest.parquet')
        held_positions = pd.read_parquet('dyn_opt/unittesting/data/optimized_positions.parquet')

        
        notional_exposure_per_contract = dyn_opt.get_notional_exposure_per_contract(unadj_prices, multipliers)
        weight_per_contract = dyn_opt.get_weight_per_contract(notional_exposure_per_contract, capital)

        fixed_cost_per_contract = 3.0
        one_day_costs = np.array(fixed_cost_per_contract)
        instrument_weight = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)
        IDM = 2.50
        maximum_forecast_ratio = 2.0
        maximum_portfolio_risk = 0.30
        maximum_jump_risk = 0.75
        maximum_position_leverage = 4.0
        maximum_correlation_risk = 0.65
        maximum_portfolio_leverage = 20.0
        max_acceptable_pct_of_open_interest = 0.01
        max_forecast_buffer = 0.50
        cost_penalty_scalar = 10

        # only use last 500 positions (everything else should take care of itself in the code)
        ideal_positions = ideal_positions[-500:]

        df = dyn_opt.single_day_optimized_positions(
            covariances_one_day=covariances.iloc[-1],
            jump_covariances_one_day=jump_covariances.iloc[-1],
            held_positions_one_day=held_positions.iloc[-2], # -2 because we want to use the previous day's positions
            ideal_positions_one_day=ideal_positions.iloc[-1],
            weight_per_contract_one_day=weight_per_contract.iloc[-1],
            costs_per_contract_one_day=one_day_costs,
            notional_exposure_per_contract_one_day=notional_exposure_per_contract.iloc[-1],
            open_interest_one_day=open_interest.iloc[-1],
            instrument_weight_one_day=instrument_weight.iloc[-1],
            tau=tau,
            capital=capital,
            IDM=IDM,
            maximum_forecast_ratio=maximum_forecast_ratio,
            maximum_position_leverage=maximum_position_leverage,
            max_acceptable_pct_of_open_interest=max_acceptable_pct_of_open_interest,
            max_forecast_buffer=max_forecast_buffer,
            maximum_portfolio_leverage=maximum_portfolio_leverage,
            maximum_correlation_risk=maximum_correlation_risk,
            maximum_portfolio_risk=maximum_portfolio_risk,
            maximum_jump_risk=maximum_jump_risk,
            asymmetric_risk_buffer=asymmetric_risk_buffer,
            cost_penalty_scalar=cost_penalty_scalar,
            additional_data=(ideal_positions.columns, ideal_positions.index[-1])
        )

        # Only optimized for last 500 values
        expected_df = pd.read_parquet('dyn_opt/unittesting/data/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        df.name = expected_df.iloc[-1].name # name is the date

        pd.testing.assert_series_equal(df, expected_df.iloc[-1])

if __name__ == '__main__':
    unittest.main(failfast=True)