from flask import Flask, request, render_template
import sqlite3
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

# Paths to databases
nifty50_db_path = 'nifty50_data_v1.db'
prediction_db_path = 'predictions/predictions.db'

# Output directory
output_html_path = 'charts/candlestick_charts.html'

# Function to load and process data from a database
def load_and_process_data(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        # Load data from the table
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Drop duplicate rows based on the Datetime column, keeping the last occurrence
    data = data.drop_duplicates(subset=['Datetime'], keep='last')

    # Sort data by Datetime to maintain order
    data['Datetime'] = pd.to_datetime(data['Datetime'])  # Ensure Datetime is in datetime format
    data = data.sort_values(by='Datetime')

    # Select the last 120 rows
    return data.tail(120)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nifty50_table = request.form["nifty50_table"]
        prediction_table = request.form["prediction_table"]

        # Load and process data from nifty50_data_v1.db
        nifty50_data = load_and_process_data(nifty50_db_path, nifty50_table)

        # Load and process data from predictions.db
        prediction_data = load_and_process_data(prediction_db_path, prediction_table)

        # Generate candlestick chart for nifty50_data
        fig1 = go.Figure(data=[go.Candlestick(
            x=nifty50_data['Datetime'],
            open=nifty50_data['Open'],
            high=nifty50_data['High'],
            low=nifty50_data['Low'],
            close=nifty50_data['Close']
        )])
        fig1.update_layout(title='Nifty50 Data Candlestick Chart', xaxis_title='Datetime', yaxis_title='Price')

        # Generate candlestick chart for predictions
        fig2 = go.Figure(data=[go.Candlestick(
            x=prediction_data['Datetime'],
            open=prediction_data['Open'],
            high=prediction_data['High'],
            low=prediction_data['Low'],
            close=prediction_data['Close']
        )])
        fig2.update_layout(title='Prediction Data Candlestick Chart', xaxis_title='Datetime', yaxis_title='Price')

        # Save the charts as HTML
        with open(output_html_path, 'w') as f:
            f.write('<h1>Nifty50 Data Candlestick Chart</h1>')
            f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<h1>Prediction Data Candlestick Chart</h1>')
            f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))

        return f"HTML file with candlestick charts has been updated: {output_html_path}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
