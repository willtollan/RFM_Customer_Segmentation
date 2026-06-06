import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configure layout to fit data tables comfortably
st.set_page_config(page_title="Machine Learning App", layout="wide")

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model!')

# 1. Define separate cached functions to load each dataset type
@st.cache_data
def load_raw_data(file_path):
    # Reads a 1,000-row sample of the heavy Excel file to preserve RAM performance
    return pd.read_excel(file_path, nrows=1000)

@st.cache_data
def load_preprocessed_data(file_path):
    # Reads the full preprocessed CSV file exported from your notebook
    return pd.read_csv(file_path)


# 2. Main data display component
with st.expander('Data Inspection Workspace', expanded=True):
    
    # --- Raw Data Section ---
    st.subheader('Raw Data')
    st.write('This is a preview (first 1000 rows) of the original transaction dataset from the source Excel file:')
    
    try:
        df_raw = load_raw_data('data/online_retail_II.xlsx')
        st.dataframe(df_raw)
    except FileNotFoundError:
        st.error("Could not find 'data/online_retail_II.xlsx'. Please check your repository file path.")

    st.markdown("---") # Visual divider line

    # --- Preprocessed Data Section ---
    st.subheader('Preprocessed Data')
    st.write('This is the fully aggregated, cleaned, and outlier-filtered RFM feature dataset:')
    
    try:
        df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
        st.dataframe(df_preprocessed)
        st.metric(label="Total Unique Customers Scored", value=len(df_preprocessed))
    except FileNotFoundError:
        st.error("Could not find 'data/preprocessed_data.csv'. Please check your repository file path.")

