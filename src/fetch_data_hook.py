import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv()

BASE_DIR = '../SQL/'
host = os.environ['host']
database = os.environ['database']
user = os.environ['user']
password = os.environ['password']
port = os.environ['port']

def fetch_sql_file(filename: str):
    try:
        # Create engine
        engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

        # Read SQL file
        with open(BASE_DIR + filename + '.sql', 'r') as file:
            sql_query = file.read()

        # Establish connection and execute query
        with engine.connect() as connection:
            # Execute the complex query
            result = connection.execute(text(sql_query))

            # Convert result to DataFrame manually
            output_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        return output_df

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fetch_sql_code(sql_query: str):
    try:
        # Create the engine
        engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

        # Wrap the raw SQL query in `text()`
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            output_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        return output_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
