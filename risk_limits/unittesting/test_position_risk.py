import sys

# Add parent directory to path
sys.path.append('../risk_management')

import unittest
import pandas as pd
import numpy as np

from risk_limits.position_risk import (
    max_leverage_position_limit, max_forecast_position_limit,
    max_pct_of_open_interest_position_limit, position_limit_aggregator
)

class TestPortfolioRisk(unittest.TestCase):
    def test_max_leverage_position_limit_1(self):
        position = 2.0
        max_leverage = 2.0
        capital=100
        exposure=100

        result = max_leverage_position_limit(max_leverage, capital, exposure, position)

        self.assertEqual(result, 2.0)
    
    def test_max_leverage_position_limit_2(self):
        position = 1.0
        max_leverage = 1.0
        capital=100
        exposure=200

        result = max_leverage_position_limit(max_leverage, capital, exposure, position)

        self.assertEqual(result, 0.5)

    def test_max_forecast_position_limit_1(self):
        position = 2.0
        capital=100
        IDM=2.0
        tau=0.2
        maximum_forecast_ratio=2.0
        max_forecast_buffer=0.1
        notional_exposure_per_contract=100
        annualized_volatility=0.2
        instrument_weight=0.5

        result = max_forecast_position_limit(
            maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight,
            notional_exposure_per_contract, annualized_volatility, position)

        self.assertEqual(result, 2.0)
    
    def test_max_forecast_position_limit_2(self):
        position = 5.0
        capital=100
        IDM=2.0
        tau=0.2
        maximum_forecast_ratio=2.0
        max_forecast_buffer=0.1
        notional_exposure_per_contract=100
        annualized_volatility=0.2
        instrument_weight=0.5

        result = max_forecast_position_limit(
            maximum_forecast_ratio, capital, IDM, tau, max_forecast_buffer, instrument_weight,
            notional_exposure_per_contract, annualized_volatility, position)

        self.assertEqual(result, 2.2)
    
    def test_max_pct_of_open_interest_position_limit_1(self):
        position = 2.0
        max_acceptable_pct_of_open_interest=0.1
        open_interest=100

        result = max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest, open_interest, position)

        self.assertEqual(result, 2.0)

    def test_max_pct_of_open_interest_position_limit_2(self):
        position = 25
        max_acceptable_pct_of_open_interest=0.1
        open_interest=100

        result = max_pct_of_open_interest_position_limit(max_acceptable_pct_of_open_interest, open_interest, position)

        self.assertEqual(result, 10)

    def test_position_limit_aggregator_1(self):
        max_leverage = 2.0
        capital=100
        exposure=100
        IDM=2.0
        tau=0.2
        maximum_forecast_ratio=2.0
        max_forecast_buffer=0.1
        annualized_volatility=0.2
        instrument_weight=0.5
        max_acceptable_pct_of_open_interest=0.1
        open_interest=100
        position=2.0

        result = position_limit_aggregator(
            max_leverage, capital, IDM, tau, maximum_forecast_ratio, max_acceptable_pct_of_open_interest, max_forecast_buffer, 
            position, exposure, annualized_volatility, instrument_weight, open_interest)

        self.assertEqual(result, 2.0)

    def test_position_limit_aggregator_2(self):
        max_leverage = 2.0
        capital=100
        exposure=100
        IDM=2.0
        tau=0.05
        maximum_forecast_ratio=2.0
        max_forecast_buffer=0.1
        annualized_volatility=0.2
        instrument_weight=0.5
        max_acceptable_pct_of_open_interest=0.1
        open_interest=100
        position=100

        result = position_limit_aggregator(
            max_leverage, capital, IDM, tau, maximum_forecast_ratio, max_acceptable_pct_of_open_interest, max_forecast_buffer, 
            position, exposure, annualized_volatility, instrument_weight, open_interest)

        self.assertEqual(result, 0.55)

if __name__ == '__main__':
    unittest.main(failfast=True)