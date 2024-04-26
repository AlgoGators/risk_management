import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd

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
            maximum_jump_risk)

        # Only optimized for last 500 values
        expected_df = pd.read_parquet('dyn_opt/unittesting/data/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        pd.testing.assert_frame_equal(df, expected_df)

if __name__ == '__main__':
    unittest.main(failfast=True)