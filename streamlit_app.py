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
    actual_df = pd.read_sql(f"SELECT * FROM {selected_table} ORDER BY Datetime DESC LIMIT 60;", actual_conn)
    pred_df = pd.read_sql(f"SELECT * FROM {selected_table}_predictions ORDER BY Datetime DESC LIMIT 60;", pred_conn)

    actual_conn.close()
    pred_conn.close()

    # Convert datetime columns to datetime format
    actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)
    pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)

    # Drop duplicate entries in the 'Datetime' column
    actual_df = actual_df.drop_duplicates(subset=['Datetime'])
    pred_df = pred_df.drop_duplicates(subset=['Datetime'])

    # Filter data to include only between 9:15 AM to 3:30 PM
    actual_df = actual_df[(actual_df['Datetime'].dt.time >= pd.to_datetime("09:15:00").time()) &
                           (actual_df['Datetime'].dt.time <= pd.to_datetime("15:30:00").time())]

    pred_df = pred_df[(pred_df['Datetime'].dt.time >= pd.to_datetime("09:15:00").time()) &
                       (pred_df['Datetime'].dt.time <= pd.to_datetime("15:30:00").time())]

    # Plot the candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=actual_df['Datetime'],
        open=actual_df['Open'],
        high=actual_df['High'],
        low=actual_df['Low'],
        close=actual_df['Close'],
        name='Actual Data',
        increasing_line_color='green',  # Color for actual data (increasing)
        decreasing_line_color='red',  # Color for actual data (decreasing)
        increasing_fillcolor='rgba(0,255,0,0.2)',  # Color fill for actual data (increasing)
        decreasing_fillcolor='rgba(255,0,0,0.2)',  # Color fill for actual data (decreasing)
    )])

    # Add predictions to the chart
    fig.add_trace(go.Candlestick(
        x=pred_df['Datetime'],
        open=pred_df['Predicted_Open'],  # Use correct column names from prediction table
        high=pred_df['Predicted_High'],
        low=pred_df['Predicted_Low'],
        close=pred_df['Predicted_Close'],
        name='Predicted Data',
        increasing_line_color='blue',  # Color for predicted data (increasing)
        decreasing_line_color='orange',  # Color for predicted data (decreasing)
        increasing_fillcolor='rgba(0,0,255,0.2)',  # Color fill for predicted data (increasing)
        decreasing_fillcolor='rgba(255,165,0,0.2)',  # Color fill for predicted data (decreasing)
    ))

    # Update layout for better visuals
    fig.update_layout(
        title=f"Candlestick Chart for {selected_table}",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        width=1200,  # Increased width of the chart
        height=600,  # Adjusted height of the chart
        xaxis=dict(
            tickformat='%H:%M',  # Format the x-axis to show time (HH:MM)
            tickangle=45,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True
        ),
    )

    st.plotly_chart(fig)

# Load and display the data when a table is selected
if selected_table:
    load_and_plot_data(selected_table)
