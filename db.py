import streamlit as st
import pyodbc
import pandas as pd

# Initialize connection.
# Uses st.experimental_singleton to only run once.
# @st.cache_resource
# def init_connection():
#    return pyodbc.connect(
#        "DRIVER={ODBC Driver 17 for SQL Server}"
#        + ";SERVER=" + st.secrets["db_server"]
#        + ";DATABASE=" + st.secrets["db_database"]
#        + ";UID=" + st.secrets["db_username"]
#        + ";PWD=" + st.secrets["db_password"]
#    )


def get_data(jumlah_row=19999):
    dt = pd.read_csv('lps_data.csv', sep='|')
    df = dt.head(jumlah_row).fillna(method='bfill')
    return df
