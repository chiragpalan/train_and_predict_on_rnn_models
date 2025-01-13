import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Function to fetch table names from the database
def fetch_table_names(db_path):
    conn = sqlite3.connect(db_path)
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    conn.close()
    return tables

# Fetch available tables from both actual and predicted databases
actual_db_path = 'nifty50_data_v1.db'
pred_db_path = 'predictions/predictions.db'
actual_tables = fetch_table_names(actual_db_path)
pred_tables = fetch_table_names(pred_db_path)

# Combine actual and predicted table names for selection
table_options = list(set(actual_tables) & set([t.replace('_predictions', '') for t in pred_tables]))

# Create the dropdown menu for table selection
selected_table = st.selectbox("Select Table", table_options)

# Function to load the selected table's data and plot the candlestick chart
def load_and_plot_data(selected_table):
    # Connect to the databases
    actual_conn = sqlite3.connect(actual_db_path)
    pred_conn = sqlite3.connect(pred_db_path)

    # Load the actual and predicted data based on selected table
    actual_df = pd.read_sql(f"SELECT * FROM {selected_table};", actual_conn)
    pred_df = pd.read_sql(f"SELECT * FROM {selected_table}_predictions;", pred_conn)

    actual_conn.close()
    pred_conn.close()

    # Convert datetime columns to datetime format
    actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)
    pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)

    # Merge dataframes on Datetime
    joined_df = pd.merge(actual_df, pred_df, on='Datetime', how='inner')

    # Drop duplicates and keep the last instances
    joined_df.drop_duplicates(subset=['Datetime'], inplace=True, keep='last')

    # Get the last 60 rows
    last_60_rows = joined_df.tail(60)

    # Plot the candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=last_60_rows['Datetime'],
        open=last_60_rows['Open'],
        high=last_60_rows['High'],
        low=last_60_rows['Low'],
        close=last_60_rows['Close'],
        name='Actual Data'
    )])

    # Add predictions to the chart
    fig.add_trace(go.Candlestick(
        x=last_60_rows['Datetime'],
        open=last_60_rows['Open_pred'],
        high=last_60_rows['High_pred'],
        low=last_60_rows['Low_pred'],
        close=last_60_rows['Close_pred'],
        name='Predicted Data'
    ))

    fig.update_layout(
        title=f"Candlestick Chart for {selected_table}",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

# Load and display the data when a table is selected
if selected_table:
    load_and_plot_data(selected_table)
