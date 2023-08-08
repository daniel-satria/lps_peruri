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
import util as utl
import base64

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score

from sklearn.model_selection import train_test_split

import pickle

import warnings
warnings.simplefilter(action='ignore')


# Page configuration
st.set_page_config(page_title="True AI",
                   page_icon="assets/paques-favicon.ico", layout="wide",
                   )

# Loading CSS
utl.local_css("assets/custom.css")


@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .appview-container > section[tabindex="0"] {
        background: bottom center/contain no-repeat url("data:image/png;base64,%s");
        background-color: #182136;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('assets/bg.png')


# Logo in side bar configuration
st.sidebar.image("assets/paques-navbar-logo.png",
                 output_format='PNG')

# Sidebar Menu
with st.sidebar:
    menu_selected = option_menu("", ["Home", "Data Exploration", "Data Editing", "Feature Engineering", "Modelling"],
                                icons=["house", "card-list", "pencil-square",
                                       "columns-gap", "gear"],
                                menu_icon="cast",
                                default_index=0,
                                styles={
                                    "container": {"padding": "0!important", "background-color": "#2E3D63"},
                                    # "icon": {"color": "orange", "font-size": "25px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444"},
                                    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
    })

# Configuring home menu
if menu_selected == "Home":
    # st.write("Welcome")
    st.image("assets/trueai-header.png",
             output_format='PNG')

# Configuring data exploration menu
if menu_selected == "Data Exploration":

    st.markdown("<h2 class='menu-title'>Data Exploration</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Analyzing and visualizing a dataset to gain a deeper understanding of its characteristics, structure, and potential patterns</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    # Setting the upload variabel
    uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                     type="csv",
                                     help="The supported file is only in csv formatted",
                                     )

    # Setting the upload options when there's file on uploader menu
    if uploaded_file is not None:
        try:
            # Uploading Dataframe
            dataframe = get_data(uploaded_file)

            # Storing dataframe to session state
            st.session_state["uploaded_file"] = dataframe

        except:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

    # Showing the uploaded file from session state
    try:
        st.write(st.session_state.uploaded_file)
        st.success("The data have been successfully uploaded")

        # Initiating pandas profiling
        if st.button('Plot the Data Exploration'):
            pr = st.session_state.uploaded_file.profile_report()
            st_profile_report(pr)

        else:
            st.write("")
    except:
        st.markdown("<span class='info-box'>Please upload any data</span>",
                    unsafe_allow_html=True)

    st.write("")

# Configuring Data Editing Menu
if menu_selected == "Data Editing":

    st.markdown("<h2 class='menu-title'>Data Editing</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Reviewing, cleaning, and modifying the dataset to address various data quality issues before using it to train a machine learning model</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    # Bringing back the data from uploaded_file session_state
    if "uploaded_file" in st.session_state:

        # Assigning uploaded_file in session state to a variable
        dataframe = st.session_state.uploaded_file

        # Initiating data on session state
        if "data" not in st.session_state:
            st.session_state.data = dataframe

        st.markdown("<h3 class='menu-secondary'>Original Data</h3>",
                    unsafe_allow_html=True)
        st.write(dataframe)
        st.write(":green[Data Shape :]", dataframe.shape)

    else:

        # Upload data variable if there is no data uploaded_file in session state
        uploaded_file = st.file_uploader("Choose a file to upload for exploring",
                                         type="csv",
                                         help="The supported file is only in csv formatted",
                                         )

        # Confiuring uploaded data
        if uploaded_file is not None:

            # Uploading Dataframe
            dataframe = get_data(uploaded_file)

            st.write(dataframe)
            st.write(dataframe.shape)

            st.session_state.uploaded_file = dataframe

    if "uploaded_file" not in st.session_state:
        st.markdown("<span class='info-box'>Please upload any data</span>",
                    unsafe_allow_html=True)

    else:

        if st.checkbox('Edit Data'):

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

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            st.write("Please click the data on x to delete the record.")

            # Initating Data Editor
            edited_data = st.data_editor(
                modified_df,
                key="data_editor",
                on_change=callback,
                hide_index=False,
                column_config=column_config,
            )
            st.write(":green[Edited Data Shape, :]",
                     edited_data.drop(columns=['x'], axis=1).shape)

            if "edited_data" not in st.session_state:
                st.session_state["edited_data"] = edited_data.drop(columns=[
                                                                   'x'], axis=1)
            else:
                st.session_state["edited_data"] = edited_data.drop(columns=[
                                                                   'x'], axis=1)

        else:
            st.write("")


# Configuring Feature Engineering Menu
if menu_selected == "Feature Engineering":

    st.markdown("<h2 class='menu-title'>Feature Engineering</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Transforming raw data into a structured and usable format for training machine learning</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    if 'feature_data' not in st.session_state:
        st.session_state['feature_data'] = ""

    if 'target_data' not in st.session_state:
        st.session_state['target_data'] = ""

    if 'feature_data_train' not in st.session_state:
        st.session_state['feature_data_train'] = ""

    if 'feature_data_test' not in st.session_state:
        st.session_state['feature_data_test'] = ""

    if 'scaled_data_train' not in st.session_state:
        st.session_state['scaled_data_train'] = ""

    if 'scaled_data_test' not in st.session_state:
        st.session_state['scaled_data_test'] = ""

    if 'y_train' not in st.session_state:
        st.session_state['y_train'] = ""

    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = ""

    # Making task option menu for feature engineering
    task_selected = option_menu("", ["Feature Engineering for Classification/Regression",
                                     "Feature Engineering for Clustering"],
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

    # Setting engineering for Classification/Regression
    if task_selected == "Feature Engineering for Classification/Regression":

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Assigning upload file variable
        uploaded_file = st.file_uploader("Choose a file to upload for training data",
                                         type="csv",
                                         help="The supported file is only in csv formatted",
                                         )
        # Configuring uploaded data
        if uploaded_file is not None:

            # Uploading Dataframe
            dataframe = get_data(uploaded_file)

            # Initiating data on session state
            if "data" not in st.session_state:
                st.session_state['data'] = dataframe

        else:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

        # Menu if data already stored in session state for classification/regression
        if 'data' in st.session_state:

            pilihan_kolom = list(st.session_state.data.columns)

            # Making column for selecting feature and target
            col1, col2 = st.columns(2)

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            # Assigning option for feature column
            with col1:
                st.markdown("<br>", unsafe_allow_html=True)
                feature_column = st.multiselect("Select any column to be featured for Classification/Regression",
                                                st.session_state.data.columns,
                                                default=list(
                                                    st.session_state.data.columns),
                                                placeholder="Select columns")

            # Assigning option for target column
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                target_column = st.selectbox("Select column to be the target",
                                             st.session_state.data.columns)

            # Making column for showing features and target
            col3, col4 = st.columns([3, 1])

            with col3:
                st.write("List of Feature Data")
                st.write(st.session_state.data[feature_column])
                st.write("Feature column shape : ",
                         st.session_state.data[feature_column].shape)

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

                # Assigning Scaler Object and fitting the data
                scaler = MinMaxScaler()

                # Fitting anf transforming the data
                scaled_data_train = scaler.fit_transform(X_train,
                                                         y_train)

                # Making dataframe out of scaled data train
                scaled_data_train_df = pd.DataFrame(
                    scaled_data_train)

                # Transforming data test and make it into dataframe
                scaled_data_test = scaler.transform(X_test)
                scaled_data_test_df = pd.DataFrame(scaled_data_test)

                st.success("The data have been scaled!")

                # Showing scaled data
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing Scaled Data Train
                st.markdown("<h4 class='menu-secondary'>Data Train Scaled</h3>",
                            unsafe_allow_html=True)  # edit dhanis
                st.write(scaled_data_train)
                st.write(":green[Scaled Data Train shape :]",
                         scaled_data_train.shape)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)

                # Showin Scaled Data test
                st.markdown("<h4 class='menu-secondary'>Data Test Scaled</h3>",
                            unsafe_allow_html=True)  # edit dhanis
                st.write(scaled_data_test)
                st.write(":green[Scaled Data Test shape :]",
                         scaled_data_test.shape)

                # Reassing session state to be used later
                st.session_state['scaled_data_train'] = scaled_data_train_df
                st.session_state['scaled_data_test'] = scaled_data_test_df
                st.session_state['feature_data_train'] = X_train
                st.session_state['feature_data_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test

        else:
            st.write("")

    # Option engineering for clustering
    if task_selected == "Feature Engineering for Clustering":

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)

        # Assigning upload file variable
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
                st.session_state['data'] = dataframe

        else:
            st.markdown("<span class='info-box'>Please upload any data</span>",
                        unsafe_allow_html=True)

        # Menu if data already stored in session state for clustering
        if 'data' in st.session_state:

            pilihan_kolom = list(st.session_state.data.columns)

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            # Option menu of feature column for clustering
            feature_column = st.multiselect("Select any column to be featured for Clustering",
                                            st.session_state.data.columns,
                                            default=list(
                                                st.session_state.data.columns),
                                            placeholder="Select columns")

            # Giving two spaces
            st.markdown("<br>", unsafe_allow_html=True)

            st.write("Feature Data for Clustering")
            st.write(st.session_state.data[feature_column])
            st.session_state['feature_data'] = st.session_state.data[feature_column]
            st.write(":green[The shape of the data :]",
                     st.session_state.data[feature_column].shape)

            if st.checkbox("Scale Data"):

                # Initiate scaler object
                scaler = MinMaxScaler()

                # Fitting and transforming the data
                scaled_data_train_df = pd.DataFrame(
                    scaler.fit_transform(st.session_state.feature_data))

                st.success("The data have been scaled!")

                # Giving one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing scaled data train
                st.markdown("<h4 class='menu-secondary'>Data Train Scaled</h3>",
                            unsafe_allow_html=True)  # edit dhanis
                st.write(scaled_data_train_df)
                st.write(":green[Scaled data train shape :]",
                         scaled_data_train_df.shape)

                st.session_state.scaled_data_train = scaled_data_train_df

# Configuring Modelling Menu
if menu_selected == "Modelling":

    st.markdown("<h2 class='menu-title'>Modelling</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Designing machine learning model alghorithm and its hyper-parameters</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    task_selected = option_menu("", ["Classification", "Regression", "Clustering"],
                                icons=["house", "card-list", "award"],
                                menu_icon="cast",
                                orientation="horizontal",
                                default_index=0,
                                styles={
                                    "container": {"background-color": "#2E3D63"},
                                    # "icon": {"color": "orange", "font-size": "25px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#444444", "text-align-last": "center"},
                                    "nav-link-selected": {"color": "#FF7F00", "background-color": "rgba(128, 128, 128, 0.1)"}
    })

    # Configuring Classification Task
    if task_selected == "Classification":

        if "scaled_data_train" in st.session_state:
            st.write(st.session_state.scaled_data_train)
        else:
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
                    st.markdown("<span class='info-box'>Please upload any data</span>",
                                unsafe_allow_html=True)

        # Adding one space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)  # edit dhanis

        # Selecting Model for Classification
        model_selection = st.selectbox(
            "Select Machine Learning Model for Classification Task",
            ("Logistic Regression", "Random Forest")
        )

        st.write("Model selected:", model_selection)

        # Setting Logistic Regression Model
        if model_selection == "Logistic Regression":

            col1, col2, col3 = st.columns(3)

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            with col1:
                # Setting Logistic Regression Penalty
                log_res_penalty = st.radio(
                    "Norm of the penalty",
                    ('l2', 'l1', 'none'))

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
                    min_value=0.001,
                    value=1.0,
                    step=0.01)

            # Logistic Regression Object
            log_res_obj = LogisticRegression(
                penalty=log_res_penalty, C=log_res_inverse, solver=log_res_solver)

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to Logistic Re Model"):

                # Initiating variable to fir data
                X_train = st.session_state.scaled_data_train
                X_test = st.session_state.scaled_data_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                # Fitting model to data
                log_res_obj.fit(X_train, y_train)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Predicting train data
                y_train_predict = log_res_obj.predict(X_train)
                y_train_predict_df = pd.DataFrame(y_train_predict)

                # Predicting test data
                y_test_predict = log_res_obj.predict(X_test)
                y_test_predict_df = pd.DataFrame(y_test_predict)

                # Predicting F1 score
                classification_report_train = pd.DataFrame(
                    classification_report(
                        y_train,
                        y_train_predict,
                        labels=[0, 1],
                        output_dict=True
                    )).transpose()
                classification_report_test = pd.DataFrame(
                    classification_report(
                        y_test,
                        y_test_predict,
                        labels=[0, 1],
                        output_dict=True
                    )).transpose()

                # Showing Data Real vs Prediction
                with st.expander("Show Data"):
                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                   axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                           axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Train with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_train_full_prediction)

                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                           y_test_df, y_test_predict_df],
                                                          axis=1)
                    st.markdown("<h4 class='menu-secondary'>Data Test with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_test_full_prediction)

                # Giving space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing score
                with st.expander("Show Classification Score"):
                    st.markdown("<h4 class='menu-secondary'>Train Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(classification_report_train)
                    st.markdown("<h4 class='menu-secondary'>Test Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(classification_report_test)

                # Giving two spaces
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing ROC-AUC Score
                with st.expander("Show ROC-AUC Report"):
                    st.markdown("<h5 class='menu-secondary'>Train ROC-AUC Score</h5>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(roc_auc_score(
                        y_train, log_res_obj.predict_proba(X_train)[:, 1]))
                    st.markdown("<h5 class='menu-secondary'>Test ROC-AUC Score</h5>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(roc_auc_score(
                        y_test, log_res_obj.predict_proba(X_test)[:, 1]))

                # Giving two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing Confusion Matrix
                with st.expander("Show Confusion Matrix"):

                    st.markdown("<h4 class='menu-secondary'>Confusion Matrix Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis

                    # Showing Confusion Matrix Display
                    cm = confusion_matrix(
                        y_test, y_test_predict, labels=[0, 1])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=[0, 1])
                    disp.plot()
                    st.pyplot(plt.show())

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
                    min_value=2,
                    value=100,
                    step=1)

            with col3:
                # Minimum number of samples to be at a left node
                rfc_min_samples_leaf = st.number_input(
                    "Minium Sample Leaf",
                    min_value=2,
                    step=1)

            # Random Forest Classifier Object
            rfc_obj = RandomForestClassifier(
                criterion=rfc_criterion,
                max_depth=rfc_max_depth,
                min_samples_leaf=rfc_min_samples_leaf)

            # Fitting Data to Random Forest Classifier Model
            if st.button("Fit Data to Random Forest Model"):

                # Initiating variable to fit data
                X_train = st.session_state.scaled_data_train
                X_test = st.session_state.scaled_data_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                # Fitting model to data
                rfc_obj.fit(X_train, y_train)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Predicting train data
                y_train_predict = rfc_obj.predict(X_train)
                y_train_predict_df = pd.DataFrame(y_train_predict)

                # Predicting test data
                y_test_predict = rfc_obj.predict(X_test)
                y_test_predict_df = pd.DataFrame(y_test_predict)

                # Predicting F1 score for train data
                classification_report_train = pd.DataFrame(classification_report(
                    y_train,
                    y_train_predict,
                    labels=[0, 1],
                    output_dict=True
                )).transpose()

                # Predicting F1 score for test data
                classification_report_test = pd.DataFrame(classification_report(
                    y_test,
                    y_test_predict,
                    labels=[0, 1],
                    output_dict=True
                )).transpose()

                with st.expander("Show Data"):
                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                   axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                           axis=1)

                    st.markdown("<h5 class='menu-secondary'>Data Train with Prediction</h5>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_train_full_prediction)

                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                           y_test_df, y_test_predict_df],
                                                          axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Test with Prediction</h5>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_test_full_prediction)

                # Giving space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing Classification Report Score
                with st.expander("See Classification Score"):
                    st.markdown("<h4 class='menu-secondary'>Train Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(classification_report_train)
                    st.markdown("<h4 class='menu-secondary'>Test Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(classification_report_test)

                    # Giving space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander("See ROC-AUC Report"):
                    st.write("Train ROC-AUC Score")
                    st.write(roc_auc_score(
                        y_train, rfc_obj.predict_proba(X_train)[:, 1]))
                    st.write("Train ROC-AUC Score")
                    st.write(roc_auc_score(
                        y_test, rfc_obj.predict_proba(X_test)[:, 1]))

                # Giving space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander("See Confusion Matrix"):

                    st.markdown("<h4 class='menu-secondary'>Confusion Matrix Score</h4>",
                                unsafe_allow_html=True)  # edit dhanis

                    # Showing Confusion Matrix Display
                    cm = confusion_matrix(
                        y_test, y_test_predict, labels=[0, 1])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=[0, 1])
                    disp.plot()
                    st.pyplot(plt.show())

    # Configuring Regression Task
    if task_selected == "Regression":

        if "scaled_data_train" in st.session_state:
            st.write(st.session_state.scaled_data_train)
        else:
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
                    st.markdown("<span class='info-box'>Please upload any data</span>",
                                unsafe_allow_html=True)

        # Markdown to give space
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)

        # Showing option of the Model
        model_selection = st.selectbox(
            "Select Machine Learning Model for Regression Task",
            ("Linear Regression", "SVM", "Random Forest")
        )
        st.write("Model selected:", model_selection)

        # Option if Linear Regression selected
        if model_selection == 'Linear Regression':
            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Setting Linear Regression fitting intercept
            lin_reg_fit_intercept = st.radio(
                "Calculating the intercept for the model",
                (True, False)
            )
            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Setting Linear Regression positive coefficients
            lin_reg_positive = st.radio(
                "Forcing the coefficients to be positive",
                (False, True)
            )

            # Linear Regression Object
            lin_reg_obj = LinearRegression(
                fit_intercept=lin_reg_fit_intercept,
                positive=lin_reg_positive
            )

            # Fitting Data to Logistic Regression Model
            if st.button("Fit Data to Linear Regression Model"):

                # Initiating variable to data fitting
                X_train = st.session_state.scaled_data_train
                X_test = st.session_state.scaled_data_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                # Fitting model to data
                lin_reg_obj.fit(X_train, y_train)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Predicting train data
                y_train_predict = lin_reg_obj.predict(X_train)
                y_train_predict_df = pd.DataFrame(y_train_predict)

                # Predicting test data
                y_test_predict = lin_reg_obj.predict(X_test)
                y_test_predict_df = pd.DataFrame(y_test_predict)

                # Calculating mean absolute error
                mae_train = mean_absolute_error(
                    y_train, y_train_predict)
                mae_test = mean_absolute_error(
                    y_test, y_test_predict)

                # Calculating mean squarred error
                mse_train = mean_squared_error(
                    y_train, y_train_predict)
                mse_test = mean_squared_error(
                    y_test, y_test_predict)

                with st.expander("Show Data"):
                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                   axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                           axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Train with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_train_full_prediction)

                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                           y_test_df, y_test_predict_df],
                                                          axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Test with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_test_full_prediction)

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing mean absolute score
                with st.expander("Show Mean Absolute Score"):
                    st.write("Train Score")
                    st.write(mae_train)
                    st.write("Test Score")
                    st.write(mae_test)

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing score
                with st.expander("Show Mean Squarred Score"):
                    st.write("Train Score")
                    st.write(mse_train)
                    st.write("Test Score")
                    st.write(mse_test)

                    # Giving space
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        # Option if Random Forest Regressor selected
        if model_selection == 'Random Forest':

            col1, col2, col3 = st.columns(3)

            with col1:
                # Setting Random Forest Classifier Split Criterion
                rfr_criterion = st.radio(
                    "Split Criterion",
                    ('squared_error', 'absolute_error',
                     'friedman_mse', 'poisson'))

            with col2:
                # Minimal Sample Split
                rfr_max_depth = st.number_input(
                    "Maximum Depth of the Tree",
                    min_value=2,
                    value=100,
                    step=1)

            with col3:
                # Minimum number of samples to be at a left node
                rfr_min_samples_leaf = st.number_input(
                    "Minium Sample Leaf",
                    min_value=2,
                    step=1)

            # Random Forest Regressor Object
            rfr_obj = RandomForestRegressor(
                criterion=rfr_criterion,
                max_depth=rfr_max_depth,
                min_samples_leaf=rfr_min_samples_leaf)

            # Fitting Data to Random Forest Regressor Model
            if st.button("Fit Data to Random Forest Model"):

                # Initiating variable to data fitting
                X_train = st.session_state.scaled_data_train
                X_test = st.session_state.scaled_data_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                # Fitting model to data
                rfr_obj.fit(X_train, y_train)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Predicting train data
                y_train_predict = rfr_obj.predict(X_train)
                y_train_predict_df = pd.DataFrame(y_train_predict)

                # Predicting test data
                y_test_predict = rfr_obj.predict(X_test)
                y_test_predict_df = pd.DataFrame(y_test_predict)

                # Calculating mean absolute error
                mae_train = mean_absolute_error(
                    y_train, y_train_predict)
                mae_test = mean_absolute_error(
                    y_test, y_test_predict)

                # Calculating mean squarred error
                mse_train = mean_squared_error(
                    y_train, y_train_predict)
                mse_test = mean_squared_error(
                    y_test, y_test_predict)

                with st.expander("Show Data"):
                    # Changing target series column to dataframe
                    y_train_df = y_train.to_frame()
                    y_test_df = y_test.to_frame()

                    # Concatting target actual column
                    target_actual_full = pd.concat([y_train_df, y_test_df],
                                                   axis=0)

                    # Adding index to predicttion of target train
                    train_index = list(y_train_df.index)
                    y_train_predict_df['index'] = train_index
                    y_train_predict_df.set_index('index', inplace=True)

                    # Adding index to prediction of target test
                    test_index = list(y_test_df.index)
                    y_test_predict_df['index'] = test_index
                    y_test_predict_df.set_index('index', inplace=True)

                    # Renaming target columns name of train data
                    y_train_df.columns = ["Target Actual"]
                    y_train_predict_df.columns = ["Target Predicted"]

                    # Showing data train full
                    data_train_full_prediction = pd.concat([st.session_state.feature_data_train,
                                                            y_train_df, y_train_predict_df],
                                                           axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Train with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_train_full_prediction)

                    # Renaming target columns name of test data
                    y_test_df.columns = ["Target Actual"]
                    y_test_predict_df.columns = ["Target Predicted"]

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Showing data train full
                    data_test_full_prediction = pd.concat([st.session_state.feature_data_test,
                                                           y_test_df, y_test_predict_df],
                                                          axis=1)

                    st.markdown("<h4 class='menu-secondary'>Data Test with Prediction</h4>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_test_full_prediction)

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing mean absolute score
                with st.expander("Show Mean Absolute Score"):
                    st.write("Train Score")
                    st.write(mae_train)
                    st.write("Test Score")
                    st.write(mae_test)

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Showing score
                with st.expander("Show Squarred Score"):
                    st.write("Train Score")
                    st.write(mse_train)
                    st.write("Test Score")
                    st.write(mse_test)

    # Configuring Clustering Task
    if task_selected == "Clustering":

        if "scaled_data_train" in st.session_state:
            st.write(st.session_state.scaled_data_train)
        else:
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
                    st.markdown("<span class='info-box'>Please upload any data</span>",
                                unsafe_allow_html=True)

        # Giving two spaces
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-secondary'>Model Configuration</h3>",
                    unsafe_allow_html=True)

        # Option for Clustering Model
        model_selection = st.selectbox(
            "Select Machine Learning Model for Clustering Task",
            ("K-Means", "Spectral Clustering", "DBSCAN")
        )

        st.write("Model selected:", model_selection)

        if model_selection == "K-Means":

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            # Adding column section for K-Means Hyper-parameters
            col1, col2, col3, col4 = st.columns(4)
            col5, col6, col7, col8 = st.columns(4)

            # Making variable of K-Means' hyper-parameter input
            with col1:
                algorithm = st.radio(
                    "K-Means Algorithm",
                    ("lloyd", "elkan")
                )

            with col2:
                n_clusters = st.number_input(
                    "Number of Clusters",
                    min_value=2,
                    value=3,
                    step=1
                )

            with col3:
                max_iter = st.number_input(
                    "Maximum of iterations",
                    min_value=2,
                    value=300,
                    step=1
                )

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            with col5:
                init = st.radio(
                    "Method of initialization",
                    ("k-means++", "random")
                )

            with col6:
                n_init = st.number_input(
                    "[Number of Run Different Centroid Seeds",
                    min_value=2,
                    value=10,
                    step=1
                )

            with col7:
                random_state = st.number_input(
                    "[Random state",
                    min_value=1,
                    value=555,
                    step=1
                )

            # K-Means Clustering Object
            kmeans_obj = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                algorithm=algorithm
            )

            # Fitting Data to K-Means Clustering Model
            if st.button("Fit Data to K-Means"):

                # Initiating variable to fir data
                X_train = st.session_state.scaled_data_train

                # Fitting data to model and getting clusters
                clusters = kmeans_obj.fit_predict(X_train)

                # Adding two spaces
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success("Training Success")

                # Copy original data into new variable
                data_full_clustered = st.session_state.data.copy()

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                # Added clusters into data and showing them accoridngly
                data_full_clustered['Cluster'] = clusters

                with st.expander("Show Data"):

                    st.markdown("<h4 class='menu-secondary'>Original Data with Clusters</h3>",
                                unsafe_allow_html=True)  # edit dhanis
                    st.write(data_full_clustered)

                # Adding one space
                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander("Show Evaluation Score"):

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.write("Calinski-Harabasz Index")
                    st.write(calinski_harabasz_score(X_train, clusters))

                    # Adding one space
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.write("Davies-Bouldin Index")
                    st.write(davies_bouldin_score(X_train, clusters))
