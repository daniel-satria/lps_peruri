import pandas as pd
import streamlit as st
from util import get_data
from sklearn.preprocessing import MinMaxScaler


# Bringing back the data from uploaded_file session_state
if "uploaded_file" in st.session_state:

    # Assigning uploaded_file in session state to a variable
    dataframe = st.session_state.uploaded_file

    # Showing data
    st.markdown("<h2 style='text-align: center; color: red;'>Original Data</h1>",
                unsafe_allow_html=True)
    st.write(dataframe)

else:

    # Upload data variable if there is no data uploaded_file in session state
    uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The file will be used for training",
                                     )

    # Confiuring uploaded data
    if uploaded_file is not None:

        # Uploading Dataframe
        dataframe = get_data(uploaded_file)

        st.write(dataframe)

        st.session_state.uploaded_file = dataframe

if "uploaded_file" not in st.session_state:
    st.write("Please upload any data")

else:

    if st.checkbox('Edit Data'):

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Initiating data on session state
        if "data" not in st.session_state:
            st.session_state.data = dataframe

        # Callback function to delete records in data
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

        # Configuring column to delete
        columns = st.session_state["data"].columns
        column_config = {column: st.column_config.Column(
            disabled=True) for column in columns}
        modified_df = st.session_state["data"].copy()
        modified_df["x"] = False

        # Moving delete column to be the first
        modified_df = modified_df[["x"] +
                                  modified_df.columns[:-1].tolist()]

        st.markdown("<h2 style='text-align: center; color: violet;'>Modify Data</h1>",
                    unsafe_allow_html=True)

        st.write("Please click the data on x column to delete the record.")

        # Initating Data Editor
        data_editor_variable = st.data_editor(
            modified_df,
            key="data_editor",
            on_change=callback,
            hide_index=False,
            column_config=column_config
        )

        st.write(st.session_state.data_editor)

        if "data" not in st.session_state:
            st.session_state['data'] = data_editor_variable
        else:
            st.session_state['data'] = data_editor_variable

    else:
        st.write("")
