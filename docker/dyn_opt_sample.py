import subprocess
import pickle
import pandas as pd

def start_dyn_opt_container():
    # Start dyn_opt Docker container
    subprocess.run(["docker", "run", "--rm", "dyn_opt_image"])

def prepare_dataframes():
    # Load DataFrame inputs
    unadj_prices = pd.read_parquet('dyn_opt/unittesting/data/unadj_prices.parquet')
    multipliers = pd.read_parquet('dyn_opt/unittesting/data/multipliers.parquet')
    ideal_positions = pd.read_parquet('dyn_opt/unittesting/data/ideal_positions.parquet')
    covariances = pd.read_parquet('dyn_opt/unittesting/data/covariances.parquet')
    jump_covariances = pd.read_parquet('dyn_opt/unittesting/data/jump_covariances.parquet')
    open_interest = pd.read_parquet('dyn_opt/unittesting/data/open_interest.parquet')
    return unadj_prices, multipliers, ideal_positions, covariances, jump_covariances, open_interest

def serialize_values(*args):
    # Serialize the values using pickle
    serialized_values = [pickle.dumps(arg) for arg in args]
    return serialized_values

def main():
    # Step 1: Start dyn_opt Docker container
    start_dyn_opt_container()

    # Step 2: Prepare DataFrame inputs
    unadj_prices, multipliers, ideal_positions, covariances, jump_covariances, open_interest = prepare_dataframes()

    # Other values
    capital = 500_000
    tau = 0.20
    asymmetric_risk_buffer = 0.05
    fixed_cost_per_contract = 3.0
    instrument_weight = pd.DataFrame(1 / len(ideal_positions.columns), index=ideal_positions.index, columns=ideal_positions.columns)
    IDM = 2.50
    maximum_forecast_ratio = 2.0
    maximum_portfolio_risk = 0.30
    maximum_jump_risk = 0.75
    maximum_position_leverage = 4.0
    maximum_correlation_risk = 0.65
    maximum_portfolio_leverage = 20.0
    max_acceptable_pct_of_open_interest = 0.01
    max_forecast_buffer = 0.50
    cost_penalty_scalar = 10

    # Step 3: Serialize values
    serialized_values = serialize_values(
        capital, tau, asymmetric_risk_buffer, fixed_cost_per_contract, 
        instrument_weight, IDM, maximum_forecast_ratio, maximum_portfolio_risk, 
        maximum_jump_risk, maximum_position_leverage, maximum_correlation_risk, 
        maximum_portfolio_leverage, max_acceptable_pct_of_open_interest, 
        max_forecast_buffer, cost_penalty_scalar
    )

    # Step 4: Pass serialized values to dyn_opt container
    p = subprocess.Popen(["docker", "run", "--rm", "-i", "dyn_opt_image"], stdin=subprocess.PIPE)
    for serialized_value in serialized_values:
        p.stdin.write(serialized_value)
    p.stdin.close()

    # Step 5: Serialize and pass DataFrames to dyn_opt container
    serialized_dfs = serialize_values([unadj_prices, multipliers, ideal_positions, covariances, jump_covariances, open_interest]) #[pickle.dumps(df) for df in [unadj_prices, multipliers, ideal_positions, covariances, jump_covariances, open_interest]]
    for serialized_df in serialized_dfs:
        p.stdin.write(serialized_df)
    p.stdin.close()

    #! Step 6: Receive the returned values


if __name__ == "__main__":
    main()
