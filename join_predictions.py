import sqlite3
import pandas as pd
# Paths to the databases
pred_db_path = 'predictions/predictions.db'
actual_db_path = 'nifty50_data_v1.db'
join_db_path = 'join_pred.db'

def join_tables(pred_db_path, actual_db_path, join_db_path):
    pred_conn = sqlite3.connect(pred_db_path)
    actual_conn = sqlite3.connect(actual_db_path)
    join_conn = sqlite3.connect(join_db_path)

    pred_tables = [t for t in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", pred_conn)['name'].tolist() if t != 'sqlite_sequence']
    actual_tables = [t for t in pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", actual_conn)['name'].tolist() if t != 'sqlite_sequence']

    for pred_table in pred_tables:
        actual_table = pred_table.replace('_predictions', '')
        if actual_table in actual_tables:
            pred_df = pd.read_sql(f"SELECT * FROM {pred_table};", pred_conn)
            actual_df = pd.read_sql(f"SELECT * FROM {actual_table};", actual_conn)

            pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)
            actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)

            joined_df = pd.merge(actual_df, pred_df, on='Datetime', how='inner')
            joined_df.to_sql(f'{actual_table}_joined', join_conn, if_exists='replace', index=False)

    pred_conn.close()
    actual_conn.close()
    join_conn.close()

# Join the tables and store the result in join_pred.db
join_tables(pred_db_path, actual_db_path, join_db_path)
