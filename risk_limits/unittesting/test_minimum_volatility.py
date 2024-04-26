import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd
import numpy as np

from risk_limits.minimum_volatility import minimum_volatility

class TestMinimumVolatility(unittest.TestCase):
    def test_minimum_volatility(self):
        variances = pd.read_parquet('risk_limits/unittesting/data/GARCH_variances.parquet')

        annualized_volatilies = variances ** 0.5 * (256 ** 0.5)

        max_forecast_ratio = 2.0
        IDM = 2.5
        tau = 0.2
        maximum_leverage = 2.0

        result : pd.Series = minimum_volatility(max_forecast_ratio, IDM, tau, maximum_leverage, 1/3, annualized_volatilies.iloc[5000])

        np.testing.assert_array_equal(result.values, np.array([True, False, False]))



if __name__ == '__main__':
    unittest.main()
