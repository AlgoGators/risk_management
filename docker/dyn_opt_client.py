import requests
import pandas as pd
import numpy as np
import json

def sanitize_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df.index = df.index.astype(str)  # Convert index to string
    return df.where(pd.notnull(df), None)  # Convert NaN to None

def serialize_data(data):
    try:
        return json.dumps(data)
    except Exception as e:
        print("Error serializing data:", e)
        print("Data:", data)
        return None

def client_request(data : dict) -> pd.DataFrame:
    serialized_data = serialize_data(data)
    if serialized_data is not None:
        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://localhost:5000/aggregator', data=serialized_data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            result_df = pd.read_json(result)
            return result_df
        else:
            raise Exception(f"Error: {response.status_code}")

def main():
    unadj_prices = pd.read_parquet('dyn_opt/unittesting/data/unadj_prices.parquet')
    multipliers = pd.read_parquet('dyn_opt/unittesting/data/multipliers.parquet')
    ideal_positions = pd.read_parquet('dyn_opt/unittesting/data/ideal_positions.parquet')
    covariances = pd.read_parquet('dyn_opt/unittesting/data/covariances.parquet')
    jump_covariances = pd.read_parquet('dyn_opt/unittesting/data/jump_covariances.parquet')
    open_interest = pd.read_parquet('dyn_opt/unittesting/data/open_interest.parquet')
    instrument_weight = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)

    unadj_prices = sanitize_data(unadj_prices)
    multipliers = sanitize_data(multipliers)
    ideal_positions = sanitize_data(ideal_positions)
    covariances = sanitize_data(covariances)
    jump_covariances = sanitize_data(jump_covariances)
    open_interest = sanitize_data(open_interest)
    instrument_weight = sanitize_data(instrument_weight)

    # Sample data to send in the request (replace with actual data)
    data = {
        "capital": 500_000,
        "fixed_cost_per_contract": 3.0,
        "tau": 0.2,
        "asymmetric_risk_buffer": 0.05,
        "unadj_prices": unadj_prices.to_dict(),
        "multipliers": multipliers.to_dict(),
        "ideal_positions": ideal_positions.to_dict(),
        "covariances": covariances.to_dict(),
        "jump_covariances": jump_covariances.to_dict(),
        "open_interest": open_interest.to_dict(),
        "instrument_weight": instrument_weight.to_dict(),
        "IDM": 2.5,
        "maximum_forecast_ratio": 2.0,
        "max_acceptable_pct_of_open_interest": 0.01,
        "max_forecast_buffer": 0.5,
        "maximum_position_leverage": 2.0,
        "maximum_portfolio_leverage": 20.0,
        "maximum_correlation_risk": 0.65,
        "maximum_portfolio_risk": 0.3,
        "maximum_jump_risk": 0.75,
        "cost_penalty_scalar": 10
    }

    result = client_request(data)
    print(result)