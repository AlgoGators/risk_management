import sys

# Add parent directory to path
sys.path.append('../risk-analysis')

import unittest
from risk_functions import RiskEstimates, PositionLimits, Volatility, RiskOverlay, Margins, Periods
import statistical_functions as StatisticalCalculations
from statistical_functions import Periods
import pandas as pd
import numpy as np


class TestRiskFunctions(unittest.TestCase):
    def setUp(self):
        # Initialize any objects or variables needed for the tests
        self.stat_calc = StatisticalCalculations
        self.risk_estimates = RiskEstimates()
        self.position_limits = PositionLimits()
        self.volatility = Volatility()
        self.risk_overlay = RiskOverlay()
        self.margins = Margins()

        prices_df = pd.read_csv('unittesting/test_data.csv')

        sp500_returns = []
        sp500_prices = [x for x in prices_df['SP500'].tolist() if str(x) != 'nan']

        for i in range(1, len(sp500_prices)):
            sp500_returns.append((sp500_prices[i] - sp500_prices[i-1]) / sp500_prices[i-1])
        
        dates1 = [x for x in prices_df['date1'].tolist() if str(x) != 'nan']

        self.sp500 = pd.DataFrame.from_dict({"Date" : dates1[1:],
                 "SP500" : sp500_returns})

        us10_returns = []
        us10_prices = [x for x in prices_df['US10'].tolist() if str(x) != 'nan']

        for i in range(1, len(us10_prices)):
            us10_returns.append((us10_prices[i] - us10_prices[i-1]) / us10_prices[i-1])

        dates2 = [x for x in prices_df['date2'].tolist() if str(x) != 'nan']

        self.us10 = pd.DataFrame.from_dict({"Date" : dates2[1:],
                 "US10" : us10_returns})
        
        self.returns_matrix = pd.merge(self.sp500, self.us10, on='Date', how='inner')
        self.returns_matrix = self.returns_matrix.set_index('Date')

        position_weights_dct = {'SP500' : [-2.1], 'US10' : [1.68]}

        self.position_weights = pd.DataFrame().from_dict(position_weights_dct)


    def test_SMA(self):
        # Test the SMA function
        lst = [1, 2, 3, 4, 5]
        span = 3
        expected_result = 3.0
        result = self.stat_calc.SMA(lst, span)
        self.assertEqual(result, expected_result)

    def test_SR(self):
        # Test the SR function
        returns = [0.06, 0.08, 0.08, 0.08, 0.08, 0.09, 0.09]
        expected_result = 8.0
        result = self.stat_calc.SR(returns)
        self.assertEqual(result, expected_result)

    def test_EWMA(self):
        # Test the EWMA function
        returns = [1.00, 1.25, 1.20, 0.90]
        expected_result = 1.0875000000000001
        result = self.stat_calc.EWMA(returns, span=7, threshold=3)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_stddev(self):
        # Test the stddev function
        returns = [1, 2, 3]
        expected_result = 1.0
        result = self.stat_calc.std(returns)
        self.assertEqual(result, expected_result)

    def test_VAR(self):
        # Test the VAR function
        returns = [1, 2, 3]
        expected_result = 1.0
        result = self.stat_calc.VAR(returns)
        self.assertEqual(result, expected_result)

    def test_ew_stddev(self):
        # Test the exponentialy_weighted_stddev function
        returns = [100, 100, 100, 115, 100, 100, 115, 100, 100, 115, 100, 100, 115, 100, 100, 115, 100, 100, 115]
        expected_result = 6.784005252999681
        result = self.stat_calc.exponentially_weighted_stddev(returns, span=7)
        self.assertEqual(result, expected_result)

    # def test_ew_var(self):
    #     # Test the exponentialy_weighted_var function
    #     returns = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    #     expected_result = 0.0
    #     result = self.stat_calc.exponentially_weighted_var(returns, alpha=0.94)
    #     self.assertAlmostEqual(result, expected_result)

    # def test_ew_covar(self):
    #     # Test the exponentialy_weighted_covar function
    #     returns_X = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    #     returns_Y = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    #     expected_result = 0.0
    #     result = self.stat_calc.exponentially_weighted_covar(returns_X, returns_Y, alpha=0.94)
    #     self.assertAlmostEqual(result, expected_result)

    # def test_ew_corr(self):
    #     # Test the exponentialy_weighted_corr function
    #     returns_dict = {'x' : [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #                     'y' : [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
    #     returns_df = pd.DataFrame.from_dict(returns_dict)

    #     expected_result = np.array([[1.0, np.nan], [np.nan, 1.0]])
    #     result = self.stat_calc.exponentially_weighted_correlation_matrix(returns_df, period=Periods.WEEKLY, span=7)
    #     np.testing.assert_array_equal(result, expected_result)
    
    # def test_ew_portfolio_cov(self):
    #     # Test the exponentialy_weighted_portfolio_covar function
    #     pct_returns_dict = {'x' : [1.0, 2.0, -4.0, 5.0, .5, -2.0, -1.0, -3.0, -4.0, 5.0, .5, -2.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, -4.0, 5.0, -2.0, -1.0],
    #                         'y' : [3.0, -4.0, 5.0, -2.0, -1.0, 1.0, 2.0, -3.0, -4.0, 5.0, .5, -2.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, -2.0, -1.0, 1.0, 2.0]}
    #     pct_returns_df = pd.DataFrame.from_dict(pct_returns_dict)

    #     expected_result = 0.0
    #     result = self.stat_calc.exponentially_weighted_portfolio_covar(pct_returns_df)
    #     self.assertEqual(result, expected_result)
        
    def test_carver_portfolio_correlation(self):
        # Test using Carver's example from his book (comparing SP500 and US10)
        expected_result = -0.22303378304662771

        result = self.stat_calc.correlation(self.sp500, self.us10)

        # Need this since the GitHub workflow machine is not as precise as my machine
        self.assertAlmostEqual(result, expected_result, places=5)


    def test_carver_portfolio_correlation_matrix(self):
        # Test using Carver's example from his book (comparing SP500 and US10)
        expected_result = np.array([[1.0, -0.23635242796983627], [-0.23635242796983627, 1.0]])

        result = self.stat_calc.correlation_matrix(self.returns_matrix, period=Periods.WEEKLY, window=52)

        np.testing.assert_array_equal(result, expected_result)

    def test_carver_portfolio_covar(self):
        # Test using Carver's example from his book (comparing SP500 and US10)
        expected_result = np.array([[0.0471353343383189, -0.0038835214482042125], [-0.0038835214482042125, 0.005727758363078177]])

        result = self.stat_calc.portfolio_covar(self.returns_matrix)

        np.testing.assert_array_equal(result, expected_result)


    def test_carver_portfolio_stddev(self):
        # Test using Carver's example
        expected_result = 0.5014329237041253

        result = self.stat_calc.portfolio_stddev(self.position_weights, self.returns_matrix) 

        self.assertEqual(result, expected_result)

    def test_estimated_portfolio_risk_multiplier(self):
        # Test using Carver's example
        expected_result = 0.5982854053217644

        result = self.risk_overlay.estimated_portfolio_risk_multiplier(self.position_weights, self.returns_matrix, 0.30)

        self.assertEqual(result, expected_result)

    def test_jump_risk_multiplier(self):
        expected_result = 0.6785805908551636

        result = self.risk_overlay.jump_risk_multiplier(self.position_weights, self.returns_matrix)

        self.assertEqual(result, expected_result)

    def test_correlation_risk_multiplier(self):
        expected_result = 1

        result = self.risk_overlay.correlation_risk_multiplier(self.position_weights, self.returns_matrix)

        self.assertEqual(result, expected_result)

    def test_leverage_risk_multiplier(self):
        expected_result = 1.0

        result = self.risk_overlay.leverage_risk_multiplier(self.position_weights)

        self.assertEqual(result, expected_result)

    def test_final_risk_multiplier(self):
        expected_result = 0.5982854053217644

        result = self.risk_overlay.final_risk_multiplier(self.position_weights, self.returns_matrix)

        self.assertEqual(result, expected_result)

    def test_minimum_volatility(self):
        result = self.volatility.minimum_volatility(1.5, 0.10, 0.20, self.sp500['SP500'].tolist(), 4.0)

        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main(failfast=True)
