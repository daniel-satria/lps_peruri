import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score


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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def show_roc_auc_score_binary_class(y, y_predict_proba):
    return roc_auc_score(y, y_predict_proba[1])


def show_roc_auc_score_multi_class(y, y_predict_proba):
    return roc_auc_score(y, y_predict_proba, multi_class='ovr')


# Function to plot confusion matrix
def plot_confusion_matrix_multi_class(y_test, y_test_predict, label_target):
    st.markdown("<h4 class='menu-secondary'>Confusion Matrix Score</h4>",
                unsafe_allow_html=True)

    cm = confusion_matrix(
        y_test, y_test_predict, labels=label_target)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_target)
    disp.plot()
    st.pyplot(plt.show())
