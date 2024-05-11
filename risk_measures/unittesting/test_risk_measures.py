import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd

from risk_measures import risk_measures

class TestRiskMetrics(unittest.TestCase):
    def setUp(self):
        self.trend_tables = {}
        
        instruments = ['ES_data', 'ZN_data', 'RB_data']

        for instrument in instruments:
            self.trend_tables[instrument] = pd.read_parquet(f'risk_measures/unittesting/data/{instrument}.parquet')

        self.risk_calculations = risk_measures.RiskMeasures(self.trend_tables, (0.01, 0.01, 0.98), 100)
        self.risk_calculations.construct()

    def test_risk_calculations(self):
        self.assertIsInstance(self.risk_calculations, risk_measures.RiskMeasures)
        self.assertIsInstance(self.risk_calculations.daily_returns, pd.DataFrame)
        self.assertIsInstance(self.risk_calculations.product_returns, pd.DataFrame)
        self.assertIsInstance(self.risk_calculations.GARCH_variances, pd.DataFrame)
        self.assertIsInstance(self.risk_calculations.GARCH_covariances, pd.DataFrame)

    def test_daily_returns(self):
        pd.testing.assert_frame_equal(self.risk_calculations.daily_returns, pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet'))

    def test_product_returns(self):
        pd.testing.assert_frame_equal(self.risk_calculations.product_returns, pd.read_parquet('risk_measures/unittesting/data/product_returns.parquet'))

    def test_GARCH_variances(self):
        pd.testing.assert_frame_equal(self.risk_calculations.GARCH_variances, pd.read_parquet('risk_measures/unittesting/data/GARCH_variances.parquet'))

    def test_GARCH_covariances(self):
        pd.testing.assert_frame_equal(self.risk_calculations.GARCH_covariances, pd.read_parquet('risk_measures/unittesting/data/GARCH_covariances.parquet'))

    def test_calculate_value_at_risk_historical(self):
        daily_returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')

        df = risk_measures.calculate_value_at_risk_historical(daily_returns, 0.95, 100)

        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/value_at_risk_historical.parquet'))

    def test_calculate_value_at_risk_parametric(self):
        GARCH_variances = pd.read_parquet('risk_measures/unittesting/data/GARCH_variances.parquet')

        df = risk_measures.calculate_value_at_risk_parametric(GARCH_variances, 0.95)

        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/value_at_risk_parametric.parquet'))

if __name__ == '__main__':
    unittest.main(failfast=True)