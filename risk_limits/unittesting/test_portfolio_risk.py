import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd
import numpy as np

from risk_limits.portfolio_risk import (
    max_leverage_portfolio_multiplier, correlation_risk_portfolio_multiplier,
    portfolio_risk_multiplier, jump_risk_multiplier, portfolio_risk_aggregator
)

class TestPortfolioRisk(unittest.TestCase):
    def test_leverage_portfolio_multiplier_1(self):
        positions = np.array([0.5, 0.5, 0.5])
        weighted_positions = positions * np.array([0.1, 0.9, 0.2])

        result = max_leverage_portfolio_multiplier(2.0, weighted_positions)

        self.assertEqual(result, 1.0)

    def test_leverage_portfolio_multiplier_2(self):
        positions = np.array([0.5, 0.5, 0.5])
        weighted_positions = positions * np.array([1.0, 2.0, 0.2])

        result = max_leverage_portfolio_multiplier(1.0, weighted_positions)

        self.assertEqual(result, 0.625)
    
    def test_correlation_risk_portfolio_multiplier_1(self):
        positions = np.array([0.5, 0.5, 0.5])
        weighted_positions = positions * np.array([0.1, 0.9, 0.2])

        result = correlation_risk_portfolio_multiplier(2.0, weighted_positions, np.array([0.1, 0.2, 0.3]))

        self.assertEqual(result, 1.0)

    def test_correlation_risk_portfolio_multiplier_2(self):
        positions = np.array([10, 10, 10])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        result = correlation_risk_portfolio_multiplier(1.0, weighted_positions, np.array([0.5, 0.5, 0.5]))

        self.assertEqual(result, 0.25)

    def test_portfolio_risk_multiplier_1(self):
        positions = np.array([1, 1, 1])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        result = portfolio_risk_multiplier(1.0, weighted_positions, np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

        self.assertEqual(result, 1)

    def test_portfolio_risk_multiplier_2(self):
        positions = np.array([5, 1, 1])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        result = portfolio_risk_multiplier(1.0, weighted_positions, np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

        self.assertAlmostEqual(result, 0.7856742013183862)

    def test_jump_risk_multiplier_1(self):
        positions = np.array([1, 1, 1])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        result = jump_risk_multiplier(1.0, weighted_positions, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        self.assertAlmostEqual(result, 1)

    def test_jump_risk_multiplier_2(self):
        positions = np.array([5, 1, 1])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        result = jump_risk_multiplier(1.0, weighted_positions, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

        self.assertAlmostEqual(result, 0.5555555555555556)

    def test_portfolio_risk_aggregator(self):
        positions = np.array([5, 1, 1])
        weighted_positions = positions * np.array([0.25, 0.25, 0.30])

        covariance_matrix = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        jump_covariance_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        result = portfolio_risk_aggregator(
            positions, weighted_positions, covariance_matrix, jump_covariance_matrix, 2.0, 2.0, 1.0, 1.0, pd.Timestamp('2021-01-01')
        )

        np.testing.assert_array_almost_equal(result, np.array([0.491046, 0.098209, 0.098209]))

if __name__ == '__main__':
    unittest.main(failfast=True)