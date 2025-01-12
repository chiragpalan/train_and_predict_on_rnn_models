import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf

# Ensure the 'predictions' folder exists
predictions_dir = 'predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

# Assuming the database is being written here:
predictions_db_path = os.path.join(predictions_dir, 'predictions.db')


def preprocess_new_data(df):
    """
    Preprocess new data by removing duplicates and sorting by datetime.
    Args:
        df (pd.DataFrame): The raw DataFrame.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.drop_duplicates(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    print("test_data", df.head())
    return df


def create_sequences(data, input_columns, n_steps):
    """
    Create input sequences for RNN model.
    Args:
        data (pd.DataFrame): The data to create sequences from.
        input_columns (list): Columns to be used as input features.
        n_steps (int): Number of timesteps for input sequence.
    Returns:
        np.ndarray: Input sequences for prediction.
    """
    X = []
    for i in range(len(data) - n_steps):
        X.append(data[input_columns].iloc[i:i+n_steps].values)
    return np.array(X)


def save_predictions_to_db(predictions, datetimes, db_path, table_name, scaler):
    """
    Save predictions for multiple future steps.
    """
    predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2])).reshape(predictions.shape)
    
    conn = sqlite3.connect(db_path)
    
    rows = []
    for i, datetime in enumerate(datetimes):
        for step in range(predictions.shape[1]):
            rows.append({
                'Datetime': datetime + pd.Timedelta(minutes=5 * (step + 1)),
                'Predicted_Open': predictions[i, step, 0],
                'Predicted_High': predictions[i, step, 1],
                'Predicted_Low': predictions[i, step, 2],
                'Predicted_Close': predictions[i, step, 3],
                'Predicted_Volume': predictions[i, step, 4],
            })

    # Save to SQLite: If the table doesn't exist, it will be created
    df = pd.DataFrame(rows)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


def main():
    database_path = 'nifty50_data_v1.db'
    input_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    n_steps = 12
    n_future = 3

    # Open connection to the source database
    conn = sqlite3.connect(database_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    tables.remove('sqlite_sequence')  # Remove internal sqlite table

    # Iterate over each table in the database
    for table_name in tables:
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
        df = preprocess_new_data(df)

        # Define model and scaler file paths
        model_path = os.path.join('models', f'{table_name}_model.h5')
        scaler_path = os.path.join('models', f'{table_name}_scaler.pkl')

        # Check if the model and scaler exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler for {table_name} not found. Skipping...")
            continue

        # Load the trained model and scaler
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Preprocess the data and create sequences
        df[input_columns] = scaler.transform(df[input_columns])
        X = create_sequences(df, input_columns, n_steps)

        # Make predictions
        predictions = model.predict(X)

        # Save predictions to the database
        save_predictions_to_db(predictions, df['Datetime'].iloc[n_steps:], predictions_db_path, f'{table_name}_predictions', scaler)

    conn.close()


if __name__ == "__main__":
    main()
