import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()

BASE_DIR = '../SQL/'
host=os.environ['host']
database=os.environ['database']
user=os.environ['user']
password=os.environ['password']
port=os.environ['port']

def fetch_sql_file(filename:str):
    try:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
        with open(BASE_DIR + filename + '.sql', 'r') as file:
            sql_query = file.read()

        output_df = pd.read_sql(sql_query, engine)
        return output_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fetch_sql_code(sql_query:str):
    try:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
        output_df = pd.read_sql(sql_query, engine)
        return output_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")