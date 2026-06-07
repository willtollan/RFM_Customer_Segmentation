import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configure layout to fit wide data tables comfortably
st.set_page_config(page_title="Machine Learning App", layout="wide")

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model!')

# 1. Define separate cached functions to load each dataset type
@st.cache_data
def load_raw_data(file_path):
    return pd.read_excel(file_path, nrows=1000)

@st.cache_data
def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_labeled_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_centroids_data(file_path):
    return pd.read_csv(file_path)


# 2. Data Inspection Workspace Component
with st.expander('Data Inspection Workspace', expanded=False):
    
    # --- Raw Data Section ---
    st.subheader('Raw Data')
    st.write('This is a preview (first 1000 rows) of the original transaction dataset from the source Excel file:')
    
    try:
        df_raw = load_raw_data('data/online_retail_II.xlsx')
        st.dataframe(df_raw)
    except FileNotFoundError:
        st.error("Could not find 'data/online_retail_II.xlsx'. Please check your repository file path.")

    st.markdown("---") 

    # --- Preprocessed Data Section ---
    st.subheader('Preprocessed Data')
    st.write('This is the fully aggregated, cleaned, and outlier-filtered RFM feature dataset:')
    
    try:
        df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
        st.dataframe(df_preprocessed)
        st.metric(label="Total Unique Customers", value=len(df_preprocessed))
    except FileNotFoundError:
        st.error("Could not find 'data/preprocessed_data.csv'. Please check your repository file path.")

    st.markdown("---")

    # --- Preprocessed Data with Labels Section ---
    st.subheader('Preprocessed Data with Labels')
    st.write('This is the feature dataset including the target label index and label name for classification:')
    
    try:
        df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
        st.dataframe(df_labeled)
        st.metric(label="Total Unique Labelled Customers", value=len(df_labeled))
        
        # --- Train/Test Split Note ---
        st.info("💡 **Modeling Note:** Prior to training, an **80% training and 20% testing split** was performed on this dataset. The split utilised **random shuffling** to remove structural order bias and **stratification** to strictly preserve original class balances across subsets.")
        
    except FileNotFoundError:
        st.error("Could not find 'data/preprocessed_labelled_data.csv'. Please check your repository file path.")


# 3. KMeans Clustering Results and Visualisations Component
with st.expander('KMeans Clustering Results and Visualisations', expanded=True):
    
    # --- KMeans Centroids Section ---
    st.subheader('KMeans Centroids')
    st.write('This table displays the calculated cluster centers (centroids) for each customer segment:')
    
    try:
        df_centroids = load_centroids_data('data/customer_centroids.csv')
        st.dataframe(df_centroids)
    except FileNotFoundError:
        st.error("Could not find 'data/customer_centroids.csv'. Please check your repository file path.")
        
    st.markdown("---")
    
    # --- 3D Scatter Plot Section ---
    st.subheader('KMeans Clusters 3D Scatter Plot given Features: Recency, Frequency and Monetary Value')
    st.write('Visual spatial separation of your customer segments across the three RFM dimensions:')
    
    # Create 3 columns (Left empty, Center for image, Right empty) to center and shrink the plot
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            # Removed use_container_width and specified a clean, locked width profile
            st.image('images/KMeans_clusters.png', width=600)
        except FileNotFoundError:
            st.error("Could not find 'images/KMeans_clusters.png'. Please check your repository folder path.")
        
    st.markdown("---")
    
    # --- Violin Plots Section ---
    st.subheader('Cluster Violin Plots by Feature')
    st.write('Distribution spread and density of Recency, Frequency, and Monetary Value across each cluster:')
    
    # Reusing columns to bound the maximum display boundary of the violin plots
    v_col1, v_col2, v_col3 = st.columns([1, 2, 1])
    with v_col2:
        try:
            st.image('images/cluster_violinplot_by_features.png', width=600)
        except FileNotFoundError:
            st.error("Could not find 'images/cluster_violinplot_by_features.png'. Please check your repository folder path.")



