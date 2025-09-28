import requests
import pandas as pd
import io
import psycopg2
import os

# NCEI API base URL
API_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
params = {
    "dataset": "argo",
    "bbox": "-20,40,10,80",
    "startDate": "2023-01-01",
    "endDate": "2023-03-31",
    "format": "csv"
}

# PostgreSQL connection details
DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_PORT = "5433"

def process_data_from_api():
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data_io = io.StringIO(response.text)
        df = pd.read_csv(data_io)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def to_postgresql(df, table_name="argo_data"):
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT  # <-- Add this line
        )
        cur = conn.cursor()
        # Ensure the table exists with the correct columns
        cols = ', '.join(f'"{col}"' for col in df.columns)
        insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({", ".join(["%s"] * len(df.columns))})'
        data_to_insert = [tuple(x) for x in df.to_numpy()]
        cur.executemany(insert_sql, data_to_insert)
        conn.commit()
        print("Data successfully inserted into PostgreSQL.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    df = process_data_from_api()
    if df is not None:
        to_postgresql(df)