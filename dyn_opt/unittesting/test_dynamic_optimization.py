import sys

# Add parent directory to path
sys.path.append('../DYN_OPT')

import unittest
import pandas as pd

from dyn_opt import aggregator

class TestDynamicOptimization(unittest.TestCase):
    def test_dyn_opt(self):
        capital = 500_000
        tau = 0.20
        asymmetric_risk_buffer = 0.05
        unadj_prices = pd.read_parquet('unittesting/data/unadj_prices.parquet')
        multipliers = pd.read_parquet('unittesting/data/multipliers.parquet')
        ideal_positions = pd.read_parquet('unittesting/data/ideal_positions.parquet')
        covariances = pd.read_parquet('unittesting/data/covariances.parquet')
        fixed_cost_per_contract = 3.0

        # only use last 500 positions (everything else should take care of itself in the code)
        ideal_positions = ideal_positions[-500:]

        df = aggregator(
            capital,
            fixed_cost_per_contract,
            tau,
            asymmetric_risk_buffer,
            unadj_prices,
            multipliers,
            ideal_positions,
            covariances)

        # Only optimized for last 500 values
        expected_df = pd.read_parquet('unittesting/data/optimized_positions.parquet')
        expected_df.index = pd.to_datetime(expected_df.index)

        pd.testing.assert_frame_equal(df, expected_df)

if __name__ == '__main__':
    unittest.main(failfast=True)