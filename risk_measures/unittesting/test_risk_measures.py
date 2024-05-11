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


    def test_calculate_daily_returns(self):
        df = risk_measures.calculate_daily_returns(self.trend_tables, 'Unadj_Close', 'Delivery Month', 'Date', fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet'))

    def test_calculate_weekly_returns(self):
        daily_returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')

        df = risk_measures.calculate_weekly_returns(daily_returns, fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/weekly_returns.parquet'))

    def test_calculate_product_returns(self):
        weekly_returns = pd.read_parquet('risk_measures/unittesting/data/weekly_returns.parquet')

        df = risk_measures.calculate_product_returns(weekly_returns, fill=True)

        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/product_returns.parquet'))

    def test_update_product_returns(self):
        all_product_returns = pd.read_parquet('risk_measures/unittesting/data/product_returns.parquet')
        product_returns = all_product_returns.iloc[:-1, :]
        weekly_returns = pd.read_parquet('risk_measures/unittesting/data/weekly_returns.parquet').iloc[[-1]]

        product_returns : pd.DataFrame = risk_measures.update_product_returns(product_returns, weekly_returns)
        pd.testing.assert_frame_equal(product_returns, all_product_returns)

    def test_calculate_GARCH_variances(self):
        daily_returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')

        df = risk_measures.calculate_GARCH_variances(daily_returns, 100, (0.01, 0.01, 0.98), fill=True)
        
        pd.testing.assert_frame_equal(df, pd.read_parquet('risk_measures/unittesting/data/GARCH_variances.parquet'))

    def test_update_GARCH_variance(self):
        all_variances = pd.read_parquet('risk_measures/unittesting/data/GARCH_variances.parquet')
        variances = all_variances.iloc[:-1, :] # .iloc[:-1] to simulate not having the last estimate
        returns = pd.read_parquet('risk_measures/unittesting/data/daily_returns.parquet')

        variances : pd.DataFrame = risk_measures.update_GARCH_variance(variances, returns, 100, (0.01, 0.01, 0.98))
        pd.testing.assert_series_equal(variances.iloc[-1], all_variances.iloc[-1])

    def test_calculate_GARCH_covariances(self):
        product_returns = pd.read_parquet('risk_measures/unittesting/data/product_returns.parquet')
        expected_df = pd.read_parquet('risk_measures/unittesting/data/GARCH_covariances.parquet')

        df = risk_measures.calculate_GARCH_covariances(product_returns, 100, (0.01, 0.01, 0.98), fill=True)

        pd.testing.assert_frame_equal(df, expected_df)

    def test_update_GARCH_covariance(self):
        all_covariances = pd.read_parquet('risk_measures/unittesting/data/GARCH_covariances.parquet')
        covariances = all_covariances.iloc[:-1, :]
        returns = pd.read_parquet('risk_measures/unittesting/data/product_returns.parquet')

        covariances : pd.DataFrame = risk_measures.update_GARCH_covariance(covariances, returns, 100, (0.01, 0.01, 0.98))
        pd.testing.assert_series_equal(covariances.iloc[-1], all_covariances.iloc[-1])

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