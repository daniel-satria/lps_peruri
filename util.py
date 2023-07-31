import pandas as pd
import streamlit as st


# Caching data for dataframe
@st.cache_data
def get_data(X):
    df = pd.read_csv(X)
    return df


@st.cache_data
def callback():
    edited_rows = st.session_state["data_editor"]["edited_rows"]
    rows_to_delete = []

    for idx, value in edited_rows.items():
        if value["x"] is True:
            rows_to_delete.append(idx)

    st.session_state["data"] = (
        st.session_state["data"].drop(
            rows_to_delete, axis=0).reset_index(drop=True)
    )
