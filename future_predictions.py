import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import xgboost as xgb
import requests 
 
# Paths and configurations
DATA_DB = 'joined_data.db'
PREDICTIONS_DB = 'future_predictions.db'
MODELS_DIR = 'models_v1'
GITHUB_REPO_URL = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'

# Ensure the directory for models exists
os.makedirs(MODELS_DIR, exist_ok=True)

def download_database():
    response = requests.get(GITHUB_REPO_URL)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        print("Database downloaded successfully.")
    else:
        raise Exception("Failed to download the database.")

def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def clean_data(X, dates):
    """Remove rows with NaN or infinite values from a NumPy array and keep data aligned."""
    mask = np.isfinite(X).all(axis=1)  # Identify rows without NaN or inf
    X_clean = X[mask]
    dates_clean = dates[mask].reset_index(drop=True)
    return X_clean, dates_clean

def random_forest_predictions(model, X_scaled):
    all_preds = [est.predict(X_scaled) for est in model.estimators_ if hasattr(est, 'predict')]
    p5, p95 = np.percentile(all_preds, 5, axis=0), np.percentile(all_preds, 95, axis=0)
    main_prediction = model.predict(X_scaled)
    return main_prediction, p5, p95

def xgboost_predictions(model, X_scaled):
    dmatrix = xgb.DMatrix(X_scaled)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    individual_tree_contributions = contributions[:, :-1]
    individual_predictions = np.cumsum(individual_tree_contributions, axis=1)
    p5 = np.percentile(individual_predictions, 5, axis=1)
    p95 = np.percentile(individual_predictions, 95, axis=1)
    main_prediction = model.predict(X_scaled)
    return main_prediction, p5, p95

def gradient_boosting_predictions(model, X_scaled):
    n_estimators = model.n_estimators
    individual_predictions = np.zeros((X_scaled.shape[0], n_estimators))
    for i in range(n_estimators):
        individual_predictions[:, i] = model.estimators_[i, 0].predict(X_scaled)
    p5 = np.percentile(individual_predictions, 5, axis=1)
    p95 = np.percentile(individual_predictions, 95, axis=1)
    main_prediction = model.predict(X_scaled)
    return main_prediction, p5, p95

def save_predictions_to_db(predictions_df, prediction_table_name):
    try:
        conn = sqlite3.connect(PREDICTIONS_DB)
        predictions_df.to_sql(prediction_table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Predictions successfully saved to table: {prediction_table_name}")
    except Exception as e:
        print(f"Error while saving to database: {e}")

def process_table_for_future_predictions(table):
    df = load_data_from_table(DATA_DB, table)
    
    # Select rows where 'target_n7d' is missing
    df_missing_target = df[df['target_n7d'].isna()]
    
    if df_missing_target.empty:
        print(f"No missing target_n7d values in table {table}. Skipping.")
        return

    print(f"Processing table {table} with missing target_n7d values...")

    # Drop the target_n7d column as it's not needed for predictions
    df_missing_target = df_missing_target.drop(columns=['target_n7d'], errors='ignore')

    # Check if 'Date' column is present
    if 'Date' not in df_missing_target.columns:
        raise KeyError("The 'Date' column is missing from the data.")
    
    # Convert the 'Date' column to datetime for sorting and filtering
    df_missing_target['Date'] = pd.to_datetime(df_missing_target['Date'])

    # Filter the data to keep only the rows with the latest date
    latest_date = df_missing_target['Date'].max()
    df_latest = df_missing_target[df_missing_target['Date'] == latest_date]

    # Separate features (X) and dates
    X = df_latest.drop(columns=['Date'], errors='ignore')
    dates = df_latest['Date']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clean the data by removing rows with NaN or inf in X
    X_scaled, dates = clean_data(X_scaled, dates)

    # Use the model to make predictions
    model_types = ['random_forest', 'gradient_boosting', 'xgboost']
    prediction_functions = {
        'random_forest': random_forest_predictions,
        'gradient_boosting': gradient_boosting_predictions,
        'xgboost': xgboost_predictions
    }

    # Create an empty DataFrame to store combined predictions
    combined_predictions = pd.DataFrame({'Date': dates})

    for model_type in model_types:
        model_path = os.path.join(MODELS_DIR, f"{table}_{model_type}.joblib")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        prediction_func = prediction_functions[model_type]

        try:
            main_prediction, p5, p95 = prediction_func(model, X_scaled)

            # Add predictions to the combined DataFrame
            combined_predictions[f'Predicted_{model_type}'] = main_prediction
            combined_predictions[f'5th_Percentile_{model_type}'] = p5
            combined_predictions[f'95th_Percentile_{model_type}'] = p95

        except Exception as e:
            print(f"Error predicting with {model_type}: {e}")

    # Save the combined predictions DataFrame to the database
    prediction_table_name = f"future_prediction_{table}"
    save_predictions_to_db(combined_predictions, prediction_table_name)

def main():
    # Ensure the future_predictions.db database is created
    os.makedirs(os.path.dirname(PREDICTIONS_DB), exist_ok=True)
    
    # Download the database from GitHub
    download_database()

    tables = get_table_names(DATA_DB)

    for table in tables:
        try:
            process_table_for_future_predictions(table)
        except Exception as e:
            print(f"Error processing table {table}: {e}")

    if os.path.exists(PREDICTIONS_DB):
        print(f"Future predictions database created at: {PREDICTIONS_DB}")
    else:
        print("Future predictions database was not created.")

if __name__ == "__main__":
    main()
