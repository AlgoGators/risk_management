import pandas as pd
from _utils import client_request, convert_to_dataframes, SERVICE_TYPE

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

    result = client_request(data, 5001, SERVICE_TYPE.RISK_MEASURES)
    
    if result is None:
        raise Exception("Error: No result returned")
    
    dfs = convert_to_dataframes(result)
    print(dfs['daily_returns'])
    print(dfs['product_returns'])
    print(dfs['GARCH_variances'])
    print(dfs['GARCH_covariances'])

if __name__ == '__main__':
    main()
