import subprocess
import pickle
import pandas as pd

def start_risk_measures_container():
    subprocess.run(["docker", "run", "--rm", "risk_measures_image"])

def serialize_values(*args):
    return [pickle.dumps(arg) for arg in args]

def main():
    start_risk_measures_container()

    trend_tables = {pd.read_parquet(f'unittesting/{contract}_data.parquet') for contract in ['ES', 'RB', 'ZN']}
    serialized_tables = serialize_values(trend_tables)

    
    p = subprocess.Popen(["docker", "run", "--rm", "-i", "dyn_opt_image"], stdin=subprocess.PIPE)
    for serialized_value in serialized_tables:
        p.stdin.write(serialized_value)
    p.stdin.close()

    #! Receive the values it returns
    