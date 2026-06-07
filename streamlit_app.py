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

# Input features
with st.sidebar:
  st.header('Input features')
  MonetaryValue = st.slider('MonetaryValue', 5, 4000, 1634)
  Frequency = st.slider('Frequency', 1, 12, 1)
  Recency = st.slider('Recency', 0, 375, 113)
  
  # Create a DataFrame for the input features
  data = {'MonetaryValue': MonetaryValue,
          'Frequency': Frequency,
          'Recency': Recency}
  input_df = pd.DataFrame(data, index=[0])


# ----------------------------------------------------
# 5. DYNAMIC LIVE CUSTOMER INFERENCE ENGINE
# ----------------------------------------------------

# Main display expander block on the dashboard
with st.expander('🔮 Dynamic Customer Segmentation Classifier', expanded=True):
    st.subheader('Live Inference Panel')
    st.write('Adjust the features in the left sidebar to classify a customer profile in real time.')
    
    try:
        # Load the model directly here to prevent root-level scoping failure blocks
        import joblib
        model_path = 'models/random_forest_model.pkl'
        
        # Load model using Streamlit's resource caching
        @st.cache_resource
        def inline_load_model(path):
            return joblib.load(path)
            
        active_model = inline_load_model(model_path)
        
        # Display the active input features dataframe passed from your sidebar controls
        st.write('**Active User Input Features Matrix (Passed to Model):**')
        st.dataframe(input_df)
        
        # --- STRIP EXTRA FIELDS AND ENSURE PERFECT FEATURE ALIGNMENT ---
        training_features = ['MonetaryValue', 'Frequency', 'Recency']
        query_features = input_df[training_features]
        
        # --- Run Real-time Machine Learning Inference ---
        raw_hard_pred = active_model.predict(query_features)
        raw_soft_pred = active_model.predict_proba(query_features)
        
        # --- HANDLE MULTI-OUTPUT OR SINGLE-OUTPUT MATRIX CONSTRAINTS DYNAMICALLY ---
        # This checks if scikit-learn returned a list of arrays (Multi-Output structure)
        if isinstance(raw_soft_pred, list):
            # Extract the positive class probability (index 1 of the first sample row) for each target array
            extracted_probs = []
            for cluster_array in raw_soft_pred:
                # cluster_array shape is (1, 2) -> [[prob_negative, prob_positive]]
                prob_positive = float(cluster_array[0][1])
                extracted_probs.append(prob_positive)
            
            # Safe Normalization step to guarantee the array values sum to exactly 1.0 (100%)
            sum_probs = sum(extracted_probs)
            if sum_probs > 0:
                probabilities = [p / sum_probs for p in extracted_probs]
            else:
                probabilities = [0.25, 0.25, 0.25, 0.25] # Fallback uniform distribution
        else:
            # Standard single-target 4-class classification array matrix layout profile
            probabilities = [float(x) for x in raw_soft_pred[0]]
            
        # Isolate final index targets safely
        if isinstance(raw_hard_pred, np.ndarray) and raw_hard_pred.ndim > 1:
            final_hard_pred = int(raw_hard_pred[0][0]) # Multi-output fallback array index choice
        else:
            final_hard_pred = int(raw_hard_pred.item())
        
        st.markdown("---")
        
        # Map cluster indexes to text labels
        cluster_mapping = {0: "Retain", 1: "Reward", 2: "Nurture", 3: "Re-Engage"}
        predicted_label = cluster_mapping.get(final_hard_pred, f"Cluster {final_hard_pred}")
        
        # --- Display Hard Prediction Outcome ---
        st.markdown(f"### **Hard Prediction Outcome:** `Cluster {final_hard_pred}: {predicted_label}`")
        
        # --- Display Soft Prediction Probabilities ---
        st.markdown("### **Soft Prediction Probabilities:**")
        prob_col1, prob_col2, prob_col3, prob_col4 = st.columns(4)
        
        # Pull values and deploy with defensive boundary clamps for st.progress tracking safety
        with prob_col1:
            val_p0 = min(max(probabilities[0], 0.0), 1.0)
            st.metric(label="🔵 Cluster 0 (Retain)", value=f"{val_p0:.1%}")
            st.progress(val_p0)
            
        with prob_col2:
            val_p1 = min(max(probabilities[1], 0.0), 1.0)
            st.metric(label="🔴 Cluster 1 (Reward)", value=f"{val_p1:.1%}")
            st.progress(val_p1)
            
        with prob_col3:
            val_p2 = min(max(probabilities[2], 0.0), 1.0)
            st.metric(label="🟢 Cluster 2 (Nurture)", value=f"{val_p2:.1%}")
            st.progress(val_p2)
            
        with prob_col4:
            val_p3 = min(max(probabilities[3], 0.0), 1.0)
            st.metric(label="🟠 Cluster 3 (Re-Engage)", value=f"{val_p3:.1%}")
            st.progress(val_p3)
            
    except ModuleNotFoundError as e:
        st.error(f"❌ **Missing Dependency:** {str(e)}. Please add `joblib` or `scikit-learn` to your `requirements.txt` file.")
    except Exception as e:
        st.error("❌ **Prediction Engine Loading Failure Details:**")
        st.warning(f"System Message: {str(e)}")





