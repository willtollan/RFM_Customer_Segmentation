import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Required to load your serialized .pkl model file

# Configure layout to fit wide data tables comfortably
st.set_page_config(page_title="Machine Learning App", layout="wide")

st.title('🤖 Machine Learning App')
st.info('This app processes transaction data, analyzes customer cohorts, and deploys a live customer classification engine.')

# 1. Define separate cached functions to load datasets and model artifacts
@st.cache_data
def load_raw_data(file_path):
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

@st.cache_resource  # Used cache_resource because a model is a persistent object
def load_serialized_model(file_path):
    return joblib.load(file_path)


# --- Load Base Files (Cached) ---
try:
    df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
    # Load model from the models folder path
    loaded_model = load_serialized_model('models/random_forest_model.pkl')
except FileNotFoundError as e:
    st.error(f"Initialization mismatch: {e}. Ensure all files exist in your repository branches.")


# 2. Data Inspection Workspace Component
with st.expander('Data Inspection Workspace', expanded=False):
    st.subheader('Raw Data')
    st.write('This is a preview (first 1000 rows) of the original transaction dataset from the source Excel file:')
    try:
        df_raw = load_raw_data('data/online_retail_II.xlsx')
        st.dataframe(df_raw)
    except FileNotFoundError:
        st.error("Could not find 'data/online_retail_II.xlsx'.")

    st.markdown("---") 

    st.subheader('Preprocessed Data')
    st.write('This is the fully aggregated, cleaned, and outlier-filtered RFM feature dataset:')
    try:
        st.dataframe(df_preprocessed)
        st.metric(label="Total Unique Customers", value=len(df_preprocessed))
    except NameError:
        st.error("Preprocessed dataframe not initialized.")

    st.markdown("---")

    st.subheader('Preprocessed Data with Labels')
    st.write('This is the feature dataset including the target label index and label name for classification:')
    try:
        df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
        st.dataframe(df_labeled)
        st.metric(label="Total Unique Labelled Customers", value=len(df_labeled))
        st.info("💡 **Modeling Note:** Prior to training, an **80% training and 20% testing split** was performed on this dataset. The split utilised **random shuffling** to remove structural order bias and **stratification** to strictly preserve original class balances across subsets.")
    except FileNotFoundError:
        st.error("Could not find 'data/preprocessed_labelled_data.csv'.")


# 3. KMeans Clustering Results and Visualisations Component
with st.expander('KMeans Clustering Results and Visualisations', expanded=False):
    st.subheader('Cluster Reference Legend')
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    with leg_col1:
        st.markdown('<div style="padding:10px; border-left: 5px solid #1f77b4; background-color: rgba(31, 119, 180, 0.1); border-radius: 4px;"><strong>Cluster 0: Retain</strong><br><span style="color:#1f77b4; font-weight:bold;">🔵 Blue Segment</span></div>', unsafe_allow_html=True)
    with leg_col2:
        st.markdown('<div style="padding:10px; border-left: 5px solid #d62728; background-color: rgba(214, 39, 40, 0.1); border-radius: 4px;"><strong>Cluster 1: Reward</strong><br><span style="color:#d62728; font-weight:bold;">🔴 Red Segment</span></div>', unsafe_allow_html=True)
    with leg_col3:
        st.markdown('<div style="padding:10px; border-left: 5px solid #2ca02c; background-color: rgba(44, 160, 44, 0.1); border-radius: 4px;"><strong>Cluster 2: Nurture</strong><br><span style="color:#2ca02c; font-weight:bold;">🟢 Green Segment</span></div>', unsafe_allow_html=True)
    with leg_col4:
        st.markdown('<div style="padding:10px; border-left: 5px solid #ff7f0e; background-color: rgba(255, 127, 14, 0.1); border-radius: 4px;"><strong>Cluster 3: Re-Engage</strong><br><span style="color:#ff7f0e; font-weight:bold;">🟠 Orange Segment</span></div>', unsafe_allow_html=True)
                    
    st.markdown("---")
    st.subheader('KMeans Centroids')
    try:
        df_centroids = load_centroids_data('data/customer_centroids.csv')
        st.dataframe(df_centroids)
    except FileNotFoundError:
        st.error("Could not find 'data/customer_centroids.csv'.")
        
    st.markdown("---")
    st.subheader('Elbow Method: Optimal Number of Clusters (K)')
    elbow_col1, elbow_col2, elbow_col3 = st.columns([0.5, 9, 0.5])
    with elbow_col2:
        try:
            st.image('images/optimal_K_elbow_method.png', width=1100)
        except FileNotFoundError:
            st.error("Could not find 'images/optimal_K_elbow_method.png'.")
        
    st.markdown("---")
    st.subheader('KMeans Clusters 3D Scatter Plot given Features: Recency, Frequency and Monetary Value')
    col1, col2, col3 = st.columns([1.5, 5, 1.5])
    with col2:
        try:
            st.image('images/KMeans_clusters.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/KMeans_clusters.png'.")
        
    st.markdown("---")
    st.subheader('Cluster Violin Plots by Feature')
    v_col1, v_col2, v_col3 = st.columns([1.5, 5, 1.5])
    with v_col2:
        try:
            st.image('images/cluster_violinplot_by_features.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/cluster_violinplot_by_features.png'.")


# 4. Random Forest Classifier Performance Metrics Component
with st.expander('Random Forest Classifier Evaluation Metrics', expanded=False):
    st.subheader('Random Forest Best Parameters')
    param_col1, param_col2, param_col3 = st.columns([1.5, 5, 1.5])
    with param_col2:
        try:
            st.image('images/tuned_RF_best_params.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_best_params.png'.")
            
    st.markdown("---")
    st.subheader('Key Metrics')
    met_col1, met_col2, met_col3 = st.columns([1.5, 5, 1.5])
    with met_col2:
        try:
            df_rf_metrics = load_rf_metrics('data/tuned_RF_key_metrics.csv')
            st.dataframe(df_rf_metrics, use_container_width=True)
        except FileNotFoundError:
            st.error("Could not find 'data/tuned_RF_key_metrics.csv'.")
        
    st.markdown("---")
    st.subheader('Classification Report')
    rep_col1, rep_col2, rep_col3 = st.columns([1.5, 5, 1.5])
    with rep_col2:
        try:
            df_rf_report = load_rf_report('data/tuned_RF_classification_report.csv')
            st.dataframe(df_rf_report, use_container_width=True)
        except FileNotFoundError:
            st.error("Could not find 'data/tuned_RF_classification_report.csv'.")
        
    st.markdown("---")
    st.subheader('Confusion Matrix')
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col2:
        try:
            st.image('images/tuned_RF_confusion_matrix.png', width=600)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_confusion_matrix.png'.")


# 5. NEW Component: Interactive Live Customer Inference Engine
with st.expander('🔮 Dynamic Customer Segmentation Classifier', expanded=True):
    st.subheader('Live Inference Panel')
    st.write('Adjust the features below to classify a hypothetical customer profile in real time.')
    
    # Check if data and model loaded successfully before showing widgets
    if 'df_preprocessed' in locals() and 'loaded_model' in locals():
        # Setup side-by-side control widgets using columns
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            val_recency = st.slider(
                'Recency (Days since last checkout)', 
                min_value=int(df_preprocessed['Recency'].min()), 
                max_value=int(df_preprocessed['Recency'].max()), 
                value=int(df_preprocessed['Recency'].median())
            )
        with input_col2:
            val_frequency = st.slider(
                'Frequency (Total distinct orders)', 
                min_value=int(df_preprocessed['Frequency'].min()), 
                max_value=int(df_preprocessed['Frequency'].max()), 
                value=int(df_preprocessed['Frequency'].median())
            )
        with input_col3:
            val_monetary = st.slider(
                'Monetary Value (Total revenue spent £)', 
                min_value=float(df_preprocessed['MonetaryValue'].min()), 
                max_value=float(df_preprocessed['MonetaryValue'].max()), 
                value=float(df_preprocessed['MonetaryValue'].median()),
                step=10.0
            )
            
        # Format input properties into a standard DataFrame matching your model's feature structure
        # Ensure the feature names match the exact column names used during model.fit()
        query_features = pd.DataFrame([{
            'Recency': val_recency,
            'Frequency': val_frequency,
            'MonetaryValue': val_monetary
        }])
        
        # --- Run Live Model Prediction ---
        hard_prediction = loaded_model.predict(query_features)[0]
        soft_prediction = loaded_model.predict_proba(query_features)[0]
        
        st.markdown("---")
        
        # Mapping numerical hard predictions back to cluster string labels


