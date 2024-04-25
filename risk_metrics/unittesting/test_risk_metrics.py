import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd
import os

from risk_metrics import risk_metrics

class TestRiskMetrics(unittest.TestCase):
    def setUp(self):
        self.trend_tables = {}
        
        instruments = ['ES_data', 'ZN_data', 'RB_data']

        for instrument in instruments:
            self.trend_tables[instrument] = pd.read_parquet(f'risk_metrics/unittesting/data/{instrument}.parquet')


    def test_calculate_daily_returns(self):
        df = risk_metrics.calculate_daily_returns(self.trend_tables, 'Unadj_Close', 'Delivery Month', 'Date', fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_metrics/unittesting/data/daily_returns.parquet'))

    def test_calculate_weekly_returns(self):
        daily_returns = pd.read_parquet('risk_metrics/unittesting/data/daily_returns.parquet')

        df = risk_metrics.calculate_weekly_returns(daily_returns, fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_metrics/unittesting/data/weekly_returns.parquet'))

    def test_calculate_product_returns(self):
        weekly_returns = pd.read_parquet('risk_metrics/unittesting/data/weekly_returns.parquet')

        df = risk_metrics.calculate_product_returns(weekly_returns, fill=True)

        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_metrics/unittesting/data/product_returns.parquet'))

    def test_calculate_GARCH_variances(self):
        daily_returns = pd.read_parquet('risk_metrics/unittesting/data/daily_returns.parquet')

        df = risk_metrics.calculate_GARCH_variances(daily_returns, 100, (0.01, 0.01, 0.98), fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_metrics/unittesting/data/GARCH_variances.parquet'))

    def test_calculate_GARCH_covariances(self):
        product_returns = pd.read_parquet('risk_metrics/unittesting/data/product_returns.parquet')

        df = risk_metrics.calculate_GARCH_covariances(product_returns, 100, (0.01, 0.01, 0.98), fill=True)

        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_metrics/unittesting/data/GARCH_covariances.parquet'))

if __name__ == '__main__':
    unittest.main(failfast=True)