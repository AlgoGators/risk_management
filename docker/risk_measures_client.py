import requests
import pandas as pd
import numpy as np
import json

def serialize_data(data):
    try:
        return json.dumps(data)
    except Exception as e:
        print("Error serializing data:", e)
        print("Data:", data)
        return None

def dataframe_to_dict(df):
    # Convert DataFrame to dictionary of dictionaries
    return {str(column): df[column].to_dict() for column in df}

def client_request(data : dict) -> pd.DataFrame:
    serialized_data = serialize_data(data)
    if serialized_data is not None:
        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://localhost:5001/risk_measures', data=serialized_data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            daily_returns = pd.read_json(result['daily_returns'])
            product_returns = pd.read_json(result['product_returns'])
            GARCH_variances = pd.read_json(result['GARCH_variances'])
            GARCH_covariances = pd.read_json(result['GARCH_covariances'])
            return {
                'daily_returns': daily_returns,
                'product_returns': product_returns,
                'GARCH_variances': GARCH_variances,
                'GARCH_covariances': GARCH_covariances
            }

        else:
            raise Exception(f"Error: {response.status_code}")

def main():
    trend_tables = {}
    for instrument in ['ES', 'RB', 'ZN']:
        trend_tables[instrument] = pd.read_parquet(f'risk_measures/unittesting/data/{instrument}_data.parquet')

    trend_tables_json = {contract: table.to_json() for contract, table in trend_tables.items()}

    data = {
        "trend_tables": trend_tables_json,
        "weights": (0.01, 0.01, 0.98),
        "warmup": 100,
        "unadj_column": "Unadj_Close",
        "expiration_column": "Delivery Month",
        "date_column": "Date",
        "fill": True
    }

    result = client_request(data)
    
    if result is not None:
        print(result['daily_returns'])
        print(result['product_returns'])
        print(result['GARCH_variances'])
        print(result['GARCH_covariances'])
        return
    raise Exception("Error: No result returned")

if __name__ == '__main__':
    main()
