import sqlite3
import pandas as pd
import os

# Ensure the 'predictions' folder exists
if not os.path.exists('predictions'):
    os.makedirs('predictions')

# Database paths
predictions_db_path = 'predictions/prediction.db'
nifty50_db_path = 'nifty50_data_v1.db'
joined_db_path = 'predictions/prediction_joined.db'


def load_data_from_db(db_path, table_name):
    """
    Load data from a SQLite database.
    Args:
        db_path (str): Path to the database file.
        table_name (str): Name of the table to load.
    Returns:
        pd.DataFrame: Data loaded from the table.
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_datetime_column(df):
    """
    Convert the 'Datetime' column to datetime format and drop timezone info.
    Args:
        df (pd.DataFrame): DataFrame containing the 'Datetime' column.
    Returns:
        pd.DataFrame: DataFrame with 'Datetime' column processed.
    """
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)  # Remove timezone info
    return df


def join_tables(predictions_df, nifty50_df):
    """
    Join the predictions and nifty50 DataFrames on the 'Datetime' column.
    Args:
        predictions_df (pd.DataFrame): Predictions data.
        nifty50_df (pd.DataFrame): Nifty50 stock data.
    Returns:
        pd.DataFrame: Joined DataFrame.
    """
    return pd.merge(predictions_df, nifty50_df, on='Datetime', how='inner')


def save_to_db(df, db_path, table_name):
    """
    Save DataFrame to SQLite database.
    Args:
        df (pd.DataFrame): DataFrame to save.
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to save to.
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()


def main():
    # Load and preprocess data from both databases
    conn = sqlite3.connect(predictions_db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    tables = [table for table in tables if table != "sqlite_sequence"]
    # tables.remove('sqlite_sequence')  # Exclude sqlite_sequence table
    conn.close()

    for table_name in tables:
        # Read data from both databases
        predictions_df = load_data_from_db(predictions_db_path, table_name)
        nifty50_df = load_data_from_db(nifty50_db_path, table_name.replace("_predictions", ""))

        # Preprocess Datetime columns
        predictions_df = preprocess_datetime_column(predictions_df)
        nifty50_df = preprocess_datetime_column(nifty50_df)

        # Join the data on 'Datetime'
        joined_df = join_tables(predictions_df, nifty50_df)

        # Save the joined data to the new database
        save_to_db(joined_df, joined_db_path, f"{table_name}_joined")

if __name__ == "__main__":
    main()
