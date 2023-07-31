import pandas as pd
import streamlit as st
from util import get_data
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(page_title="Simple AI",
                   page_icon="assets/paques-favicon.ico", layout="wide",
                   )


uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                 type="csv",
                                 help="The file will be used for training",
                                 )
# Confiuring uploaded data
if uploaded_file is not None:

    # Uploading Dataframe
    dataframe = get_data(uploaded_file)

    # Initiating data on session state
    if "data" not in st.session_state:
        st.session_state.data = dataframe


else:
    st.write("Please upload any data to edit.")


pilihan_kolom = list(st.session_state.data.columns)

col1, col2 = st.columns(2)


with col1:
    feature_column = st.multiselect("Select any column",
                                    st.session_state.data.columns,
                                    default=list(
                                        st.session_state.data.columns),
                                    placeholder="Select columns")

with col2:
    target_column = st.selectbox("Select column to be the target",
                                 st.session_state.data.columns)


col3, col4 = st.columns([3, 1])

with col3:
    st.write("List of Feature Data")
    st.write(st.session_state.data[feature_column])

with col4:
    st.write("Target Data")
    st.write(st.session_state.data[target_column])


if 'feature_data' not in st.session_state:
    st.session_state['feature_data'] = st.session_state.data[feature_column]

if 'target_data' not in st.session_state:
    st.session_state['target_data'] = st.session_state.data[target_column]

if st.button("Scale Data"):
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(st.session_state.feature_data,
                                       st.session_state.target_data)

    st.write(scaled_data)
