from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from util import get_data
from streamlit_option_menu import option_menu
import pickle


st.markdown("<h2 class='menu-title'>Load Model</h2>",
            unsafe_allow_html=True)
st.markdown("<h6 class='menu-subtitle'>Retrieving a trained machine learning model from storage and making it available to build predictions</h6>",
            unsafe_allow_html=True)
st.markdown("<hr class='menu-divider' />",
            unsafe_allow_html=True)

# Making task option menu for loading model
load_model_selected = option_menu("", ["Load Classification/Regression Model",
                                       "Load Clustering Model"],
                                  icons=["motherboard", "people"],
                                  menu_icon="cast",
                                  orientation="horizontal",
                                  default_index=0,
                                  styles={
    "container": {"background-color": "#2E3D63"},
    # "icon": {"color": "orange", "font-size": "25px"},
    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444", "text-align-last": "center"},
    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
})

# Adding one space
st.markdown("<br>", unsafe_allow_html=True)

# Loading model for classification or regression
if load_model_selected == "Load Classification/Regression Model":

    st.markdown("<h4 class='menu-secondary'>Upload Model</h3>",
                unsafe_allow_html=True)

    # Upload variable for uploading model
    uploaded_model = st.file_uploader("Choose a model to upload for making predictions",
                                      type="pkl",
                                      help="The supported file is only in pkl formatted",
                                      )

    # Setting the upload options when there's file on uploader menu
    if uploaded_model is not None:

        model = pickle.loads(uploaded_model.read())
        st.success("Model Loaded")

    else:
        st.write("")

    # Adding two spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 class='menu-secondary'>Upload File</h4>",
                unsafe_allow_html=True)

    # Upload variable for uploading file to be predicted
    uploaded_file = st.file_uploader("Choose a file to be predicted with machine learning model",
                                     type="csv",
                                     help="The supported file is only in csv formatted",
                                     )

    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)
            st.success("The data have been successfully uploaded")

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 class='menu-secondary'>Original Data</h3>",
                        unsafe_allow_html=True)

            st.write(dataframe)
            st.write("- The shape of data", dataframe.shape)
        except:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

    # Adding two spaces
    st.markdown("<br>", unsafe_allow_html=True)

# Loading model for clustering
if load_model_selected == "Load Clustering Model":

    st.markdown("<h4 class='menu-secondary'>Upload Model</h4>",
                unsafe_allow_html=True)

    # Upload variable for uploading model
    uploaded_model = st.file_uploader("Choose a model to upload for making predictions",
                                      type="pkl",
                                      help="The supported file is only in pkl formatted",
                                      )

    # Setting the upload options when there's file on uploader menu
    if uploaded_model is not None:

        model = pickle.loads(uploaded_model.read())
        st.success("Model Loaded")

    else:
        st.write("")

    # Adding three spaces
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 class='menu-secondary'>Upload File</h3>",
                unsafe_allow_html=True)

    # Upload variable for uploading file to be predicted
    uploaded_file = st.file_uploader("Choose a file to be predicted with machine learning model",
                                     type="csv",
                                     help="The supported file is only in csv formatted",
                                     )

    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)
            st.success("The data have been successfully uploaded")

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 class='menu-secondary'>Original Data</h3>",
                        unsafe_allow_html=True)
            st.write(dataframe)
            st.write("- The shape of data", dataframe.shape)

        except:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Make Prediction"):
        # Getting cluster
        clusters = model.predict(dataframe)

        # Adding cluster into data
        dataframe['Clusters'] = clusters

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 class='menu-secondary'>Original Data with Clusters</h3>",
                    unsafe_allow_html=True)
        st.write(dataframe)
        st.write("- The shape of data", dataframe.shape)
