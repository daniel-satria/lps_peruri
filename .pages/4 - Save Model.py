import streamlit as st
import pandas as pd
from util import get_data


uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                 type="csv",
                                 help="The file will be used for training",
                                 )

if uploaded_file is not None:
    try:
        # Uploading Dataframe
        dataframe = get_data(uploaded_file)

        if "data" not in st.session_state:
            st.session_state.data = dataframe
    except:
        st.write("")

if "data" not in st.session_state:
    st.session_state.data = dataframe


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


columns = st.session_state["data"].columns
column_config = {column: st.column_config.Column(
    disabled=True) for column in columns}

modified_df = st.session_state["data"].copy()
modified_df["x"] = False
# Make Delete be the first column
modified_df = modified_df[["x"] + modified_df.columns[:-1].tolist()]

st.data_editor(
    modified_df,
    key="data_editor",
    on_change=callback,
    hide_index=True,
    column_config=column_config,
)
