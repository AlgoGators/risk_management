from flask import Flask, request, jsonify
import pandas as pd
from dyn_opt import dyn_opt

app = Flask(__name__)

@app.route('/aggregator', methods=['POST'])
def aggregator():
    data = request.get_json()
    
    # Parse the input data
    capital = data['capital']
    fixed_cost_per_contract = data['fixed_cost_per_contract']
    tau = data['tau']
    asymmetric_risk_buffer = data['asymmetric_risk_buffer']
    unadj_prices = pd.DataFrame(data['unadj_prices'])
    multipliers = pd.DataFrame(data['multipliers'])
    ideal_positions = pd.DataFrame(data['ideal_positions'])
    covariances = pd.DataFrame(data['covariances'])
    jump_covariances = pd.DataFrame(data['jump_covariances'])
    open_interest = pd.DataFrame(data['open_interest'])
    instrument_weight = pd.DataFrame(data['instrument_weight'])
    IDM = data['IDM']
    maximum_forecast_ratio = data['maximum_forecast_ratio']
    max_acceptable_pct_of_open_interest = data['max_acceptable_pct_of_open_interest']
    max_forecast_buffer = data['max_forecast_buffer']
    maximum_position_leverage = data['maximum_position_leverage']
    maximum_portfolio_leverage = data['maximum_portfolio_leverage']
    maximum_correlation_risk = data['maximum_correlation_risk']
    maximum_portfolio_risk = data['maximum_portfolio_risk']
    maximum_jump_risk = data['maximum_jump_risk']
    cost_penalty_scalar = data['cost_penalty_scalar']

    positions = dyn_opt.aggregator(
        capital, fixed_cost_per_contract, tau, asymmetric_risk_buffer, 
        unadj_prices, multipliers, ideal_positions, covariances, 
        jump_covariances, open_interest, instrument_weight, IDM, 
        maximum_forecast_ratio, max_acceptable_pct_of_open_interest, 
        max_forecast_buffer, maximum_position_leverage, maximum_portfolio_leverage, 
        maximum_correlation_risk, maximum_portfolio_risk, maximum_jump_risk, 
        cost_penalty_scalar
    )

    # Convert the result to a JSON serializable format
    result_json = {
        'positions' : positions.to_json()
    }

    return jsonify(result_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
