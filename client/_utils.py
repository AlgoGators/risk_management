import pandas as pd
import numpy as np
import json
import requests
from enum import Enum

class SERVICE_TYPE(str, Enum):
    AGGREGATOR = 'aggregator'
    RISK_MEASURES = 'risk_measures'

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

def dataframe_to_dict(df):
    # Convert DataFrame to dictionary of dictionaries
    return {str(column): df[column].to_dict() for column in df}

def client_request(data : dict, port : int, service : SERVICE_TYPE) -> list[pd.DataFrame] | pd.DataFrame:
    serialized_data = serialize_data(data)
    if serialized_data is not None:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'http://localhost:{port}/{service.value}', data=serialized_data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result

        else:
            raise Exception(f"Error: {response.status_code}")
        
def convert_to_dataframes(json_dict : dict[str, str]):
    df_dict : dict[str, pd.DataFrame] = {}
    for key in json_dict.keys():
        df_dict[key] = pd.read_json(json_dict[key])
    return df_dict

        
