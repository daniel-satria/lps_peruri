import streamlit as st
import pandas as pd

df = pd.read_csv(
    r'C:\Users\daniel.satria\Desktop\Paques\Peruri\dinamic_ml\datasets\contoh.csv')

st.data_editor(df, key="data_editor")  # Test
st.write("Here's the session state:")
st.write(st.session_state["data_editor"])
