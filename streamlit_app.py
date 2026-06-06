import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model!')

# 1. Define a cached function to load the data
@st.cache_data
def load_data(file_path):
    # This heavy operation runs only once
    return pd.read_excel(file_path)

with st.expander('Data'):
    st.write('**Raw data**')
    
    # 2. Call the cached function instead of reading directly
    df = load_data('online_retail_II.xlsx')
    
    st.write(df)
