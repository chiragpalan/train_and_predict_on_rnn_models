import sqlite3
import pandas as pd
import os

def update_readme():
    db_path = 'predictions/predictions.db'
    readme_path = 'README.md'

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    readme_content = ""

    for table_name in tables:
        table_name = table_name[0]
        # Read the last 5 rows from each table
        df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY Datetime DESC LIMIT 5", conn)
        # Remove duplicates based on the Datetime column
        df = df.drop_duplicates(subset=['Datetime'])
        readme_content += f"## {table_name}\n"
        readme_content += df.to_markdown(index=False)
        readme_content += "\n\n"

    # Close the connection
    conn.close()

    # Write to README file
    with open(readme_path, 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    update_readme()
