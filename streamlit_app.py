import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Required to load your serialized .pkl model file

# Configure layout to fit wide data tables comfortably
st.set_page_config(page_title="Machine Learning App", layout="wide")

st.title('🤖 Machine Learning App')
st.info('This app processes transaction data, analyzes customer cohorts, and deploys a live customer classification engine.')

# ----------------------------------------------------
# 1. CACHED DATA & ARTIFACT LOADING FUNCTIONS
# ----------------------------------------------------
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

@st.cache_resource  # Used cache_resource because a model is a persistent object
def load_serialized_model(file_path):
    return joblib.load(file_path)


# --- Safely Initialize Base Processing Dependencies (Cached) ---
try:
    df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
    df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
    loaded_model = load_serialized_model('models/random_forest_model.pkl')
except FileNotFoundError as e:
    st.error(f"Initialization mismatch error: {e}. Please check your repository file paths.")


# ----------------------------------------------------
# 2. DATA INSPECTION WORKSPACE COMPONENT
# ----------------------------------------------------
with st.expander('Data Inspection Workspace', expanded=False):
    
    # --- Raw Data Section ---
    st.subheader('Raw Data')
    st.write('This is a preview (first 1000 rows) of the original transaction dataset from the source Excel file:')
    try:
        df_raw = load_raw_data('data/online_retail_II.xlsx')
        st.dataframe(df_raw)
    except FileNotFoundError:
        st.error("Could not find 'data/online_retail_II.xlsx'.")

    st.markdown("---") 

    # --- Preprocessed Data Section ---
    st.subheader('Preprocessed Data')
    st.write('This is the fully aggregated, cleaned, and outlier-filtered RFM feature dataset:')
    try:
        st.dataframe(df_preprocessed)
        st.metric(label="Total Unique Customers", value=len(df_preprocessed))
    except NameError:
        st.error("Preprocessed dataframe not initialized.")

    st.markdown("---")

    # --- Preprocessed Data with Labels Section ---
    st.subheader('Preprocessed Data with Labels')
    st.write('This is the feature dataset including the target label index and label name for classification:')
    try:
        st.dataframe(df_labeled)
        st.metric(label="Total Unique Labelled Customers", value=len(df_labeled))
        
        # --- Train/Test Split Note ---
        st.info("💡 **Modeling Note:** Prior to training, an **80% training and 20% testing split** was performed on this dataset. The split utilised **random shuffling** to remove structural order bias and **stratification** to strictly preserve original class balances across subsets.")
    except NameError:
        st.error("Labeled dataframe not initialized.")


# ----------------------------------------------------
# 3. KMEANS CLUSTERING RESULTS AND VISUALISATIONS
# ----------------------------------------------------
with st.expander('KMeans Clustering Results and Visualisations', expanded=False):
    
    # --- Color-Coded Legend Section ---
    st.subheader('Cluster Reference Legend')
    st.write('Use this color-coded key to identify segments across the visualizations below:')
    
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
    
    # --- KMeans Centroids Section ---
    st.subheader('KMeans Centroids')
    st.write('This table displays the calculated cluster centers (centroids) for each customer segment:')
    try:
        df_centroids = load_centroids_data('data/customer_centroids.csv')
        st.dataframe(df_centroids)
    except FileNotFoundError:
        st.error("Could not find 'data/customer_centroids.csv'.")
        
    st.markdown("---")

    # --- Elbow Method Section (Large Format Plot) ---
    st.subheader('Elbow Method: Optimal Number of Clusters (K)')
    st.write('Evaluation of Within-Cluster Sum of Squares (WCSS) to determine the mathematically optimal cluster configuration:')
    
    elbow_col1, elbow_col2, elbow_col3 = st.columns([0.5, 9, 0.5])
    with elbow_col2:
        try:
            st.image('images/optimal_K_elbow_method.png', width=1100)
        except FileNotFoundError:
            st.error("Could not find 'images/optimal_K_elbow_method.png'.")
        
    st.markdown("---")
    
    # --- 3D Scatter Plot Section (Standard Format Plot) ---
    st.subheader('KMeans Clusters 3D Scatter Plot given Features: Recency, Frequency and Monetary Value')
    st.write('Visual spatial separation of your customer segments across the three RFM dimensions:')
    
    col1, col2, col3 = st.columns([1.5, 5, 1.5])
    with col2:
        try:
            st.image('images/KMeans_clusters.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/KMeans_clusters.png'.")
        
    st.markdown("---")
    
    # --- Violin Plots Section (Standard Format Plot) ---
    st.subheader('Cluster Violin Plots by Feature')
    st.write('Distribution spread and density of Recency, Frequency, and Monetary Value across each cluster:')
    
    v_col1, v_col2, v_col3 = st.columns([1.5, 5, 1.5])
    with v_col2:
        try:
            st.image('images/cluster_violinplot_by_features.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/cluster_violinplot_by_features.png'.")


# ----------------------------------------------------
# 4. RANDOM FOREST CLASSIFIER PERFORMANCE METRICS
# ----------------------------------------------------
with st.expander('Random Forest Classifier', expanded=False):
    
    # --- Random Forest Best Parameters ---
    st.subheader('Random Forest Best Parameters')
    st.write('The optimal hyperparameters found during the grid search tuning optimization phase:')
    
    param_col1, param_col2, param_col3 = st.columns([1.5, 5, 1.5])
    with param_col2:
        try:
            st.image('images/tuned_RF_best_params.png', width=800)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_best_params.png'.")
            
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
            st.error("Could not find 'data/tuned_RF_key_metrics.csv'.")
        
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
            st.error("Could not find 'data/tuned_RF_classification_report.csv'.")
        
    st.markdown("---")
    
    # --- Confusion Matrix Section (Small Format Plot) ---
    st.subheader('Confusion Matrix')
    st.write('Matrix visualising the actual versus predicted classification distributions on test data subsets:')
    
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col2:
        try:
            st.image('images/tuned_RF_confusion_matrix.png', width=600)
        except FileNotFoundError:
            st.error("Could not find 'images/tuned_RF_confusion_matrix.png'.")


# ----------------------------------------------------
# 5. DYNAMIC LIVE CUSTOMER INFERENCE ENGINE
# ----------------------------------------------------

with st.expander('🔮 Dynamic Customer Segmentation Classifier', expanded=True):
    st.subheader('Live Inference Panel')
    st.write('Adjust the features below to classify a hypothetical customer profile in real time.')
    
    # Verify processing dependencies are active
    dependencies_loaded = 'df_preprocessed' in locals() and 'df_labeled' in locals() and 'loaded_model' in locals()
    
    if dependencies_loaded:
        # Build layout columns for features inputs
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
            
        # Structure slider configurations into an isolated DataFrame row object
        query_features = pd.DataFrame([{
            'Recency': val_recency,
            'Frequency': val_frequency,
            'MonetaryValue': val_monetary
        }])
        
        # Strip labeling parameters and map training columns order sequences
        training_features = [col for col in df_labeled.columns if col not in ['Customer ID', 'label_index', 'label_name']]
        query_features = query_features[training_features]
        
        # Run real-time machine learning inference
        raw_hard_pred = loaded_model.predict(query_features)
        raw_soft_pred = loaded_model.predict_proba(query_features)
        
        # Isolate outputs securely to strip away NumPy array formats
        final_hard_pred = int(raw_hard_pred.item())
        final_soft_pred = raw_soft_pred.flatten()
        
        st.markdown("---")
        
        # Map cluster indexes to text groupings
        cluster_mapping = {0: "Retain", 1: "Reward", 2: "Nurture", 3: "Re-Engage"}
        predicted_label = cluster_mapping.get(final_hard_pred, f"Cluster {final_hard_pred}")
        
        # Show prediction summary banner
        st.markdown(f"### **Hard Prediction Outcome:** `Cluster {final_hard_pred}: {predicted_label}`")
        
        # Show breakdown probabilities
        st.markdown("### **Soft Prediction Probabilities:**")
        prob_col1, prob_col2, prob_col3, prob_col4 = st.columns(4)
        
        with prob_col1:
            val_p0 = float(final_soft_pred.item(0))
            st.metric(label="🔵 Cluster 0 (Retain)", value=f"{val_p0:.1%}")
            st.progress(val_p0)
        with prob_col2:
            val_p1 = float(final_soft_pred.item(1))
            st.metric(label="🔴 Cluster 1 (Reward)", value=f"{val_p1:.1%}")
            st.progress(val_p1)
        with prob_col3:
            val_p2 = float(final_soft_pred.item(2))
            st.metric(label="🟢 Cluster 2 (Nurture)", value=f"{val_p2:.1%}")
            st.progress(val_p2)
        with prob_col4:
            val_p3 = float(final_soft_pred.item(3))
            st.metric(label="🟠 Cluster 3 (Re-Engage)", value=f"{val_p3:.1%}")
            st.progress(val_p3)
            
    else:
        st.warning("Prediction engine offline. Check your file deployments inside your data and models folders.")

