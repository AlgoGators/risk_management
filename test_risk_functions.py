import unittest
from risk_functions import StatisticalCalculations, RiskEstimates, PositionLimits, Volatility, RiskOverlay, Margins

class TestRiskFunctions(unittest.TestCase):
    def setUp(self):
        # Initialize any objects or variables needed for the tests
        self.stat_calc = StatisticalCalculations()
        self.risk_estimates = RiskEstimates()
        self.position_limits = PositionLimits()
        self.volatility = Volatility()
        self.risk_overlay = RiskOverlay()
        self.margins = Margins()

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
        expected_result = 0.73125
        result = self.stat_calc.EWMA(returns, span=7)
        self.assertEqual(result, expected_result)

    def test_stddev(self):
        # Test the stddev function
        returns = [1, 2, 3]
        expected_result = 1.0
        result = self.stat_calc.stddev(returns)
        self.assertEqual(result, expected_result)

    def test_ew_stddev(self):
        # Test the exponentialy_weighted_stddev function
        returns = [100, 100, 100, 115]
        expected_result = 28.2345241422361
        result = self.stat_calc.exponentially_weighted_stddev(returns, span=7)
        self.assertEqual(result, expected_result)

    def test_ew_var(self):
        # Test the exponentialy_weighted_var function
        returns = [100, 100, 100, 100]
        expected_result = 3188.7534141540527
        result = self.stat_calc.exponentially_weighted_var(returns, span=7)
        self.assertEqual(result, expected_result)
        


if __name__ == '__main__':
    unittest.main()
