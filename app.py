from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from pandas_geojson import read_geojson
import os
import auth
import db
from util import get_data
import streamlit as st
from streamlit_option_menu import option_menu
from numpy import NaN
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter(action='ignore')


# Page configuration
st.set_page_config(page_title="Simple AI",
                   page_icon="assets/paques-favicon.ico", layout="wide",
                   )

# Logo in side bar configuration
st.sidebar.image("assets/paques-navbar-logo.png",
                 output_format='PNG', width=150)

# Sidebar Menu
with st.sidebar:
    menu_selected = option_menu("Menu", ["Home", "Data Exploration", "Data Engineering", "Modelling"],
                                icons=["house", "card-list", "award", "gear"],
                                menu_icon="cast",
                                default_index=0,
                                styles={
                                "nav-link": {"font-size": "15px", "text-align": "left",
                                             "margin": "0px", "--hover-color": "#444444"}
                                })

# Configuring home menu
if menu_selected == "Home":
    st.write("Welcome")

# Configuring data exploration menu
if menu_selected == "Data Exploration":

    # Setting the upload variabel
    uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The file will be used for training the Machine Learning",
                                     )

    # Setting the upload options when there's file on uploader menu
    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)

            # Storing dataframe to session state
            st.session_state["uploaded_file"] = dataframe

        except:
            st.write("Please upload any data")

    # Showing the uploaded file from session state
    try:
        st.write(st.session_state.uploaded_file)
        st.success("The data have been successfully uploaded")

        # Initiating pandas profiling
        if st.button('Plot the Data Exploration'):
            pr = dataframe.profile_report()
            st_profile_report(pr)

        else:
            st.write("")
    except:
        st.write("")

    st.write("")

# Configuring Data Engineering Menu
if menu_selected == "Data Engineering":

    # Upload data variable
    uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The file will be used for training",
                                     )
    # Confiuring uploaded data
    if uploaded_file is not None:

        # Uploading Dataframe
        dataframe = get_data(uploaded_file)

        if st.button('Edit Data'):

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

            # Initating Data Editor
            st.data_editor(
                modified_df,
                key="data_editor",
                on_change=callback,
                hide_index=True,
                column_config=column_config,
            )

    else:
        st.write("Please upload any data to edit.")


# Configuring Modelling Menu
if menu_selected == "Modelling":

    st.markdown("<h2 style='text-align: center; color: red;'>Machine Learning Modelling Menu</h1>",
                unsafe_allow_html=True)

    task_selected = option_menu("", ["Classification", "Regression", "Clustering"],
                                icons=["house", "card-list", "award"],
                                menu_icon="cast",
                                orientation="horizontal",
                                default_index=0,
                                styles={
        "nav-link": {"font-size": "15px", "text-align": "center",
                     "margin": "0px", "--hover-color": "#444444"}
    })

    # Configuring Classification Task
    if task_selected == "Classification":

        st.markdown("<br>", unsafe_allow_html=True)

        # Setting the upload variabel
        uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                         type="csv",
                                         help="The file will be used for training the Machine Learning",
                                         )

        # Setting the upload options when there's file on uploader menu
        if uploaded_file is not None:
            try:
                # Uploading Dataframe
                dataframe = get_data(uploaded_file)

                X = dataframe.drop(columns="Outcome")
                y = dataframe["Outcome"]

                # Storing dataframe to session state
                if 'X' not in st.session_state:
                    st.session_state["X"] = X

                if 'y' not in st.session_state:
                    st.session_state["y"] = y

            except:
                st.write("Please upload any data")

        # Markdown to gice space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: cyan;'>Model Setting</h3>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Selecting Model for Classification
        model_selection = st.selectbox(
            "Select Machine Learning Model for Classification Task",
            ("Logistic Regression", "Random Forest")
        )

        st.write("Model selected:", model_selection)

        # Setting Logistic Regression Model
        if model_selection == "Logistic Regression":

            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Logistic Regression Penalty
                log_res_penalty = st.radio(
                    "Norm of the penalty",
                    ('l1', 'l2', 'None'))

            with col2:
                # Setting Logistis Regression Solver
                log_res_solver = st.radio(
                    "Algorithm optimization",
                    ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
                     ))

            with col3:
                # Inverse of regularization strength
                log_res_inverse = st.number_input(
                    "Inverse of regularization",
                    min_value=0.00001,
                    step=0.01)

            # Logistic Regression Object
            log_res_obj = LogisticRegression(
                penalty=log_res_penalty, C=log_res_inverse, solver=log_res_solver)

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to Logistic Re Model"):

                # Initiating variable to fir data
                X = st.session_state.X
                y = st.session_state.y

                log_res_obj.fit(X, y)

                st.write("Training Success")

                if st.button("Predict Data"):
                    y_predict = log_res_obj.predict(X)

                    y_predict_df = pd.DataFrame(y_predict)

                    if 'y_predict_df' not in st.session_state:
                        st.session_state['y_predict_df'] = y_predict_df

                    st.write(st.session_state.y_predict_df)

        # Setting Random Forest Classifier Model
        if model_selection == "Random Forest":

            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Random Forest Classifier Split Criterion
                rfc_criterion = st.radio(
                    "Split Criterion",
                    ('gini', 'entropy', 'log_loss'))

            with col2:
                # Minimal Sample Split
                rfc_max_depth = st.number_input(
                    "Maximum Depth of the Tree",
                    min_value=1,
                    step=1)

            with col3:
                # Minimum number of samples to be at a left node
                rfc_min_samples_leaf = st.number_input(
                    "Minium Sample Leaf",
                    min_value=2,
                    step=1)

            # Random Forest Classifier Object
            rfc_obj = RandomForestClassifier(
                criterion=rfc_criterion, max_depth=rfc_max_depth, solver=log_res_solver)

            # Fitting Data to Random Forest Classifier Model
            if st.button("Fit Data to Random Forest Model"):
                rfc_obj.fit(dataframe)

    # Configuring Classification Task
    if task_selected == "Regression":

        model_selection = st.selectbox(
            "Select Machine Learning Model for Regression Task",
            ("Linear Regression", "SVM", "Random Forest")
        )

        st.write("Model selected:", model_selection)

    # Configuring Clustering Task
    if task_selected == "Clustering":

        model_selection = st.selectbox(
            "Select Machine Learning Model for Clustering Task",
            ("K-Means", "Spectral Clustering", "DBSCAN")
        )

        st.write("Model selected:", model_selection)
