from flask import Flask, request, jsonify
from risk_measures.risk_measures import RiskMeasures
import pandas as pd

app = Flask(__name__)

@app.route('/risk_measures', methods=['POST'])
def risk_measures():
    data = request.get_json()

    # Parse the input data
    # trend_tables = {contract : pd.read_json(table) for contract, table in data['trend_tables'].items()}  #data.get('trend_tables', {})
    trend_tables = {contract: pd.read_json(table) for contract, table in data['trend_tables'].items()}

    weights = tuple(data.get('weights', (None, None, None)))
    warmup = int(data.get('warmup', None))
    unadj_column = data.get('unadj_column', None)
    expiration_column = data.get('expiration_column', None)
    date_column = data.get('date_column', None)
    fill = bool(data.get('fill', None))

    risk_measures = RiskMeasures(trend_tables, weights, warmup, unadj_column, expiration_column, date_column, fill)
    risk_measures.construct()

    daily_returns = risk_measures.daily_returns
    product_returns = risk_measures.product_returns
    GARCH_variances = risk_measures.GARCH_variances
    GARCH_covariances = risk_measures.GARCH_covariances

    # Convert the result to a JSON serializable format
    result_json = {
        'daily_returns': daily_returns.to_json(),
        'product_returns': product_returns.to_json(),
        'GARCH_variances': GARCH_variances.to_json(),
        'GARCH_covariances': GARCH_covariances.to_json()
    }

    return jsonify(result_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
