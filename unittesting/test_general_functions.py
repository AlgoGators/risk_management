import sys

# Add parent directory to path
sys.path.append('../risk_analysis')

import unittest
from general_functions import get_daily_returns
import pandas as pd
import numpy as np

class TestRiskFunctions(unittest.TestCase):
    def setUp(self):
        self.SP500_df = pd.read_csv('unittesting/_ES_Data.csv')


    def test_daily_returns(self):
        returns_df = get_daily_returns(self.SP500_df)
        
        expected_result = pd.read_csv('unittesting/_ES_Returns.csv', index_col='Date')

        returns_df.equals(expected_result)

if __name__ == '__main__':
    unittest.main(failfast=True)
