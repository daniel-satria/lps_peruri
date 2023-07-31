import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from util import get_data


if 'feature_data' not in st.session_state:
    st.session_state['feature_data'] = True

if 'target_data' not in st.session_state:
    st.session_state['target_data'] = True

if 'scaled_data_train' not in st.session_state:
    st.session_state['scaled_data_train'] = True

if 'scaled_data_test' not in st.session_state:
    st.session_state['scaled_data_test'] = True

if 'y_train' not in st.session_state:
    st.session_state['y_train'] = True

if 'y_test' not in st.session_state:
    st.session_state['y_test'] = True


uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                 type="csv",
                                 help="The file will be used for training",
                                 )
# Configuring uploaded data
if uploaded_file is not None:

    # Uploading Dataframe
    dataframe = get_data(uploaded_file)

    # Initiating data on session state
    if "data" not in st.session_state:
        st.session_state.data = dataframe

else:
    st.write("Please upload any data to edit.")

pilihan_kolom = list(st.session_state.data.columns)

# Making column for selecting feature and target
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

# Giving space
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Making column for showing features and target
col3, col4 = st.columns([3, 1])

with col3:
    st.write("List of Feature Data")
    st.write(st.session_state.data[feature_column])

with col4:
    st.write("Target Data")
    st.write(st.session_state.data[target_column])

st.session_state['feature_data'] = st.session_state.data[feature_column]
st.session_state['target_data'] = st.session_state.data[target_column]

if st.checkbox("Scale Data"):
    # Splitting data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.feature_data,
        st.session_state.target_data,
        test_size=0.25,
        random_state=555
    )

    # st.write(st.session_state.target_data)

    scaler = MinMaxScaler()

    scaled_data_train = scaler.fit_transform(X_train,
                                             y_train)

    scaled_data_train_df = pd.DataFrame(
        scaled_data_train)

    scaled_data_test = scaler.transform(X_test)

    scaled_data_test_df = pd.DataFrame(
        scaled_data_test)

    st.success("The data have been scaled!")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Data Train Scaled")
    st.write(scaled_data_train)
    st.write(scaled_data_train.shape)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Data test Scaled")
    st.write(scaled_data_test)
    st.write(scaled_data_test.shape)

    st.session_state['scaled_data_train'] = scaled_data_train_df

    st.session_state['scaled_data_test'] = scaled_data_test_df

    st.session_state['y_train'] = y_train

    st.session_state['y_test'] = y_test
