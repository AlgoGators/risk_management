import sys

# Add parent directory to path
sys.path.append('../risk_analysis')

import unittest
from risk_functions import PositionLimits, Volatility, RiskOverlay, Margins, Periods
from risk_reporting import RiskEstimates
from statistical_functions import Periods
import pandas as pd
import numpy as np


class TestRiskFunctions(unittest.TestCase):
    def setUp(self):
        # Initialize any objects or variables needed for the tests
        self.risk_estimates = RiskEstimates
        self.volatility = Volatility()

        self.margins = Margins()

        self.total_positions_df = pd.read_csv('unittesting/test_data/total_portfolio_positions.csv', index_col=0)
        self.open_interest_df = pd.read_csv('unittesting/test_data/open_interest.csv', index_col=0)
        self.notional_exposure_df = pd.read_csv('unittesting/test_data/notional_exposure.csv', index_col=0)
        self.standard_deviation_df = pd.read_csv('unittesting/test_data/standard_deviation.csv', index_col=0)

        self.capital = 500_000        

        self.position_weights = self.notional_exposure_df / self.capital

        self.position_percent_returns = pd.read_csv('unittesting/test_data/percent_returns.csv', index_col=0)

        # prices_df = pd.read_csv('unittesting/test_data.csv')

        # sp500_returns = []
        # sp500_prices = [x for x in prices_df['SP500'].tolist() if str(x) != 'nan']

        # for i in range(1, len(sp500_prices)):
        #     sp500_returns.append((sp500_prices[i] - sp500_prices[i-1]) / sp500_prices[i-1])
        
        # dates1 = [x for x in prices_df['date1'].tolist() if str(x) != 'nan']

        # self.sp500 = pd.DataFrame.from_dict({"Date" : dates1[1:],
        #          "SP500" : sp500_returns})

        # us10_returns = []
        # us10_prices = [x for x in prices_df['US10'].tolist() if str(x) != 'nan']

        # for i in range(1, len(us10_prices)):
        #     us10_returns.append((us10_prices[i] - us10_prices[i-1]) / us10_prices[i-1])

        # dates2 = [x for x in prices_df['date2'].tolist() if str(x) != 'nan']

        # self.us10 = pd.DataFrame.from_dict({"Date" : dates2[1:],
        #          "US10" : us10_returns})
        
        # self.returns_matrix = pd.merge(self.sp500, self.us10, on='Date', how='inner')
        # self.returns_matrix = self.returns_matrix.set_index('Date')

        # position_weights_dct = {'SP500' : [-2.1], 'US10' : [1.68]}

        # self.position_weights = pd.DataFrame().from_dict(position_weights_dct)

    # def test_total_portfolio_risk(self):

    def test_position_risk(self):
        print("Position Risk1")
        self.position_limits = PositionLimits(
            total_positions_df=self.total_positions_df, 
            standard_deviation_df=self.standard_deviation_df,
            notional_exposure_per_contract_df=self.notional_exposure_df,
            open_interest_df=self.open_interest_df,
            IDM=2.5,
            instrument_weight=0.10,
            risk_target=0.20,
            average_forecast=10,
            max_forecast=20,
            max_leverage_ratio=2.0,
            capital=self.capital,
            max_pct_of_open_interest=0.01,
            max_forecast_margin=0.50)

        print(self.position_limits.get_risk_adjusted_positions())
        print("Position Risk2")

    # def test_test(self):
    #     print("Test")

    def test_portfolio_risk(self):
        print("Portfolio Risk1")
        self.risk_overlay = RiskOverlay(
            total_positions_df=self.total_positions_df,
            position_weights=self.position_weights,
            position_percent_returns = self.position_percent_returns)

        print(self.risk_overlay.get_risk_adjusted_positions())
        print("Portfolio Risk2")

    # def test_estimated_portfolio_risk_multiplier(self):
    #     # Test using Carver's example
    #     expected_result = 0.5982854053217644

    #     result = self.risk_overlay.get_estimated_portfolio_risk_multiplier(self.position_weights, self.returns_matrix, 0.30)

    #     self.assertAlmostEqual(result, expected_result)

    # def test_jump_risk_multiplier(self):
    #     expected_result = 0.6785805908551636

    #     result = self.risk_overlay.get_jump_risk_multiplier(self.position_weights, self.returns_matrix)

    #     self.assertAlmostEqual(result, expected_result)

    # def test_correlation_risk_multiplier(self):
    #     expected_result = 1

    #     result = self.risk_overlay.get_correlation_risk_multiplier(self.position_weights, self.returns_matrix)

    #     self.assertAlmostEqual(result, expected_result)

    # def test_leverage_risk_multiplier(self):
    #     expected_result = 1.0

    #     result = self.risk_overlay.get_leverage_risk_multiplier(self.position_weights, 20)

    #     self.assertAlmostEqual(result, expected_result)

    # def test_final_risk_multiplier(self):
    #     expected_result = 0.5982854053217644

    #     result = self.risk_overlay.final_risk_multiplier(self.position_weights, self.returns_matrix, 20)

    #     self.assertAlmostEqual(result, expected_result)

    #!def test_minimum_volatility(self):
    #!    result = self.volatility.minimum_volatility(1.5, 0.10, 0.20, self.sp500['SP500'].tolist(), 4.0)
#!
    #!    self.assertTrue(result)
#!
    #!def test_VaR_Historical(self):
    #!    expected_result = 7895.112258922633
#!
    #!    result = self.risk_estimates.VaR_Historical(100_000, self.sp500['SP500'].tolist())
#!
    #!    self.assertAlmostEqual(result, expected_result)
#!
    #!def test_VaR_Parametrc(self):
    #!    expected_result = 6503.474483239808
#!
    #!    result = self.risk_estimates.VaR_Parametric(100_000, 0.10, 10)
#!
    #!    self.assertAlmostEqual(result, expected_result)

    # def test_max_position_forecast(self):
    #     IDM = 2.0
    #     instrument_weight = 0.10
    #     risk_target = 0.20
    #     annual_stddev = 0.011
    #     capital = 500_000
    #     scaled_forecast = 10
    #     average_forecast = 10
    #     fx_rate = 1.0
    #     price = 97
    #     multiplier = 2500

    #     notional_exposure_per_contract = price * multiplier * fx_rate

    #     number_of_contracts = scaled_forecast * capital * IDM * instrument_weight * risk_target / (average_forecast * multiplier * price * fx_rate * annual_stddev)

    #     result = self.position_limits.maximum_position_forecast(number_of_contracts, capital, IDM, instrument_weight, risk_target, notional_exposure_per_contract, annual_stddev, average_forecast, max_forecast=20)

    #     # expect the minimum of the two numbers to be the number of contracts since the max forecast is less

    #     self.assertAlmostEqual(result, number_of_contracts)

    # def test_max_position_leverage(self):
    #     fx_rate = 1.0
    #     price = 97
    #     multiplier = 2500

    #     notional_exposure_per_contract = price * multiplier * fx_rate

    #     capital = 500_000

    #     max_leverage_ratio = 2.0

    #     number_of_contracts = 50
        
    #     result = self.position_limits.maximum_position_leverage(number_of_contracts, max_leverage_ratio, capital, notional_exposure_per_contract)

    #     expected_result = 4.123711340206185

    #     self.assertAlmostEqual(result, expected_result) 

if __name__ == '__main__':
    unittest.main(failfast=True)
