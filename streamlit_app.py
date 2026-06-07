import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# 1. CACHED DATA & ARTIFACT LOADING FUNCTIONS
# ----------------------------------------------------

# Configure layout to fit wide data tables comfortably
st.set_page_config(page_title="Machine Learning App", layout="wide")

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model!')

# 1. Define separate cached functions to load each dataset type
@st.cache_data
def load_raw_data(file_path):
    # Force 'Invoice' and 'StockCode' columns to strings to prevent PyArrow rendering crashes
    return pd.read_excel(file_path, nrows=1000, dtype={'Invoice': str, 'StockCode': str})

@st.cache_data
def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_labeled_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_centroids_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_rf_metrics(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_rf_report(file_path):
    return pd.read_csv(file_path)
    
# ----------------------------------------------------
# 2. DATA INSPECTION WORKSPACE COMPONENT
# ----------------------------------------------------

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

# ----------------------------------------------------
# 3. KMEANS CLUSTERING RESULTS AND VISUALISATIONS
# ----------------------------------------------------

# 3. KMeans Clustering Results and Visualisations Component
with st.expander('KMeans Clustering Results and Visualisations', expanded=False):
    
    # --- Color-Coded Legend Section ---
    st.subheader('Cluster Reference Legend')
    st.write('Use this color-coded key to identify segments across the visualizations below:')
    
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    
    with leg_col1:
        st.markdown('<div style="padding:10px; border-left: 5px solid #1f77b4; background-color: rgba(31, 119, 180, 0.1); border-radius: 4px;">'
                    '<strong>Cluster 0: Retain</strong><br><span style="color:#1f77b4; font-weight:bold;">🔵 Blue Segment</span></div>', 
                    unsafe_allow_html=True)
                    
    with leg_col2:
        st.markdown('<div style="padding:10px; border-left: 5px solid #d62728; background-color: rgba(214, 39, 40, 0.1); border-radius: 4px;">'
                    '<strong>Cluster 1: Reward</strong><br><span style="color:#d62728; font-weight:bold;">🔴 Red Segment</span></div>', 
                    unsafe_allow_html=True)
                    
    with leg_col3:
        st.markdown('<div style="padding:10px; border-left: 5px solid #2ca02c; background-color: rgba(44, 160, 44, 0.1); border-radius: 4px;">'
                    '<strong>Cluster 2: Nurture</strong><br><span style="color:#2ca02c; font-weight:bold;">🟢 Green Segment</span></div>', 
                    unsafe_allow_html=True)
                    
    with leg_col4:
        st.markdown('<div style="padding:10px; border-left: 5px solid #ff7f0e; background-color: rgba(255, 127, 14, 0.1); border-radius: 4px;">'
                    '<strong>Cluster 3: Re-Engage</strong><br><span style="color:#ff7f0e; font-weight:bold;">🟠 Orange Segment</span></div>', 
                    unsafe_allow_html=True)
                    
    st.markdown("---")
    
    # --- KMeans Centroids Section ---
    st.subheader('KMeans Centroids')
    st.write('This table displays the calculated cluster centers (centroids) for each customer segment:')
    
    try:
        df_centroids = load_centroids_data('data/customer_centroids.csv')
        st.dataframe(df_centroids)
    except FileNotFoundError:
        st.error("Could not find 'data/customer_centroids.csv'. Please check your repository file path.")
        
    st.markdown("---")

    # --- Elbow Method Section ---
    st.subheader('Elbow Method: Optimal Number of Clusters (K)')
    st.write('Evaluation of Within-Cluster Sum of Squares (WCSS) to determine the mathematically optimal cluster configuration:')
    
    elbow_col1, elbow_col2, elbow_col3 = st.columns([0.5, 9, 0.5])
    with elbow_col2:
        try:
            st.image('images/optimal_K_elbow_method.png', width=1100)
        except FileNotFoundError:
            st.error("Could not find 'images/optimal_K_elbow_method.png'. Please check your repository folder path.")
        
    st.markdown("---")
    
    # --- 3D Scatter Plot Section ---
    st.subheader('KMeans Clusters 3D Scatter Plot given Features: Recency, Frequency and Monetary Value')
    st.write('Visual spatial separation of your customer segments across the three RFM dimensions:')
    
    col1, col2, col3 = st.columns([1.5, 5, 1.5])
    with col2:
        try:
            st.image('images/KMeans_clusters.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/KMeans_clusters.png'. Please check your repository folder path.")
        
    st.markdown("---")
    
    # --- Violin Plots Section ---
    st.subheader('Cluster Violin Plots by Feature')
    st.write('Distribution spread and density of Recency, Frequency, and Monetary Value across each cluster:')
    
    v_col1, v_col2, v_col3 = st.columns([1.5, 5, 1.5])
    with v_col2:
        try:
            st.image('images/cluster_violinplot_by_features.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/cluster_violinplot_by_features.png'. Please check your repository folder path.")

# ----------------------------------------------------
# 4. RANDOM FOREST CLASSIFIER PERFORMANCE METRICS
# ----------------------------------------------------

# 4. Random Forest Classifier Performance Metrics Component
with st.expander('Random Forest Classifier', expanded=True):
    
    # --- Random Forest Best Parameters ---
    st.subheader('Random Forest Best Parameters')
    st.write('The optimal hyperparameters found during the grid search tuning optimization phase:')
    
    param_col1, param_col2, param_col3 = st.columns([1.5, 5, 1.5])
    with param_col2:
        try:
            st.image('images/tuned_RF_best_params.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_best_params.png'. Please check your repository folder path.")
            
    st.markdown("---")
    
    # --- Key Metrics Section ---
    st.subheader('Key Metrics')
    st.write('Overall evaluation metrics for the tuned Random Forest classification model:')
    
    met_col1, met_col2, met_col3 = st.columns([1.5, 5, 1.5])
    with met_col2:
        try:
            df_rf_metrics = load_rf_metrics('data/tuned_RF_key_metrics.csv')
            st.dataframe(df_rf_metrics, use_container_width=True)
        except FileNotFoundError:
            st.error("Could not find 'data/tuned_RF_key_metrics.csv'. Please check your repository file path.")
        
    st.markdown("---")
    
    # --- Classification Report Section ---
    st.subheader('Classification Report')
    st.write('Detailed performance metrics breakdown including precision, recall, and f1-score per cluster target:')
    
    rep_col1, rep_col2, rep_col3 = st.columns([1.5, 5, 1.5])
    with rep_col2:
        try:
            df_rf_report = load_rf_report('data/tuned_RF_classification_report.csv')
            st.dataframe(df_rf_report, use_container_width=True)
        except FileNotFoundError:
            st.error("Could not find 'data/tuned_RF_classification_report.csv'. Please check your repository file path.")
        
    st.markdown("---")
    
    # --- Confusion Matrix Section ---
    st.subheader('Confusion Matrix')
    st.write('Matrix visualising the actual versus predicted classification distributions on test data subsets:')
    
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col2:
        try:
            st.image('images/tuned_RF_confusion_matrix.png', width=600)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_confusion_matrix.png'. Please check your repository folder path.")


# ----------------------------------------------------
# 5. DYNAMIC LIVE CUSTOMER INFERENCE ENGINE
# ----------------------------------------------------

# --- Sync Callbacks ---
def sync_mv_slider():
    st.session_state.mv_num = st.session_state.mv_slide

def sync_mv_num():
    st.session_state.mv_slide = st.session_state.mv_num

def sync_freq_slider():
    st.session_state.freq_num = st.session_state.freq_slide

def sync_freq_num():
    st.session_state.freq_slide = st.session_state.freq_num

def sync_rec_slider():
    st.session_state.rec_num = st.session_state.rec_slide

def sync_rec_num():
    st.session_state.rec_slide = st.session_state.rec_num


# Input features
with st.sidebar:
  st.header('Input features')
  
  # 1. MonetaryValue Sync Block
  st.number_input('MonetaryValue Input', 5, 4000, 1634, key='mv_num', on_change=sync_mv_num)
  MonetaryValue = st.slider('MonetaryValue', 5, 4000, key='mv_slide', on_change=sync_mv_slider)
  
  st.markdown("---")
  
  # 2. Frequency Sync Block
  st.number_input('Frequency Input', 1, 12, 1, key='freq_num', on_change=sync_freq_num)
  Frequency = st.slider('Frequency', 1, 12, key='freq_slide', on_change=sync_freq_slider)
  
  st.markdown("---")
  
  # 3. Recency Sync Block
  st.number_input('Recency Input', 0, 375, 113, key='rec_num', on_change=sync_rec_num)
  Recency = st.slider('Recency', 0, 375, key='rec_slide', on_change=sync_rec_slider)
  
  # Create a DataFrame for the input features
  data = {'MonetaryValue': MonetaryValue,
          'Frequency': Frequency,
          'Recency': Recency}
  input_df = pd.DataFrame(data, index=[0])




