import streamlit as st
import pandas as pd
import sqlite3
import requests
from io import BytesIO
import plotly.graph_objects as go

# GitHub raw URLs for databases
PREDICTIONS_DB_URL = "https://raw.githubusercontent.com/chiragpalan/train_and_predict_on_rnn_models/main/predictions/predictions.db"
ACTUAL_DB_URL = "https://raw.githubusercontent.com/chiragpalan/train_and_predict_on_rnn_models/main/nifty50_data_v1.db"

@st.cache_data
def download_database(url):
    """Download the database file from GitHub."""
    response = requests.get(url)
    response.raise_for_status()  # Raise error for failed requests
    return BytesIO(response.content)

def load_data_from_db(db_bytes, table_name):
    """Load data from a specific table in the database."""
    with sqlite3.connect(f"file:{db_bytes}?mode=memory&cache=shared", uri=True) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    return df

def generate_candlestick_chart(df, title):
    """Generate a candlestick chart using Plotly."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=title,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    return fig

# Streamlit app
def main():
    st.title("Real-Time Stock Market Candlestick Charts")
    st.write("Visualize actual and predicted stock data with real-time updates.")

    # Fetch the latest databases from GitHub
    st.info("Fetching the latest databases from GitHub...")
    predictions_db_bytes = download_database(PREDICTIONS_DB_URL)
    actual_db_bytes = download_database(ACTUAL_DB_URL)

    # Input table names
    pred_table = st.text_input("Enter predicted data table name:", "your_pred_table_name")
    actual_table = st.text_input("Enter actual data table name:", "your_actual_table_name")

    if st.button("Generate Charts"):
        try:
            # Load data from databases
            pred_df = load_data_from_db(predictions_db_bytes, pred_table)
            actual_df = load_data_from_db(actual_db_bytes, actual_table)

            # Clean and preprocess data
            pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)
            actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)

            pred_df.drop_duplicates(subset=['Datetime'], inplace=True, keep='last')
            actual_df.drop_duplicates(subset=['Datetime'], inplace=True, keep='last')

            pred_df.sort_values('Datetime', inplace=True)
            actual_df.sort_values('Datetime', inplace=True)

            # Get last 60 rows for visualization
            pred_df_last_60 = pred_df.tail(60)
            actual_df_last_60 = actual_df.tail(60)

            # Generate and display candlestick charts
            st.subheader("Actual Data Candlestick Chart")
            fig_actual = generate_candlestick_chart(actual_df_last_60, "Actual Data")
            st.plotly_chart(fig_actual)

            st.subheader("Predicted Data Candlestick Chart")
            fig_pred = generate_candlestick_chart(pred_df_last_60, "Predicted Data")
            st.plotly_chart(fig_pred)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
