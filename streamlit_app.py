import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Required to load your serialized .pkl model file
import shap    # Required for explainable AI local visualizations
import matplotlib.pyplot as plt

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

@st.cache_resource
def get_shap_explainer(_model):
    """Initializes the SHAP TreeExplainer from the Random Forest model."""
    if hasattr(_model, 'named_steps'):
        rf_clf = _model.named_steps['clf']
    else:
        rf_clf = _model
    return shap.TreeExplainer(rf_clf)

# --- Safely Initialize Base Processing Dependencies (Cached) ---
try:
    df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
    df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
    loaded_model = load_serialized_model('models/random_forest_model.pkl')
    explainer = get_shap_explainer(loaded_model)
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
# 5. COLLECT USER INPUTS (Sidebar State Synchronizer)
# ----------------------------------------------------

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
    data = {
        'MonetaryValue': MonetaryValue,
        'Frequency': Frequency,
        'Recency': Recency
    }
    input_df = pd.DataFrame(data, index=[0])


# ----------------------------------------------------
# 6. DYNAMIC LIVE CUSTOMER INFERENCE ENGINE & SHAP EXPLANATIONS
# ----------------------------------------------------

with st.expander('🔮 Dynamic Customer Segmentation Classifier', expanded=True):
    st.subheader('Live Inference Panel')
    st.write('Adjust the features in the left sidebar to classify a customer profile in real time.')
    
    try:
        # Enforce column sequence structure to match your exact X_train notebook layout
        training_features = ['MonetaryValue', 'Frequency', 'Recency']
        query_features = input_df[training_features]
        
        # --- EXTRACT ESTIMATOR FROM PIPELINE COHORT IF APPLICABLE ---
        if hasattr(loaded_model, 'named_steps'):
            rf_estimator = loaded_model.named_steps['clf']
        else:
            rf_estimator = loaded_model

        # --- RUN INFERENCE USING PRE-TRAINED ESTIMATOR ---
        prediction = rf_estimator.predict(query_features)
        prediction_proba = rf_estimator.predict_proba(query_features)
        
        # --- FIXED: SIMPLIFIED PROBABILITY MAPPING (MATCHES IRIS WORKING PROTOTYPE) ---
        # Converts prediction_proba directly into a clean, flat 1D array row of raw floats.
        # This completely strips layout quirks and guarantees they sum to exactly 1.0.
        raw_probabilities = np.asarray(prediction_proba).flatten()
        
        # Build DataFrame directly using the verified raw decimal vector row
        df_prediction_proba = pd.DataFrame([raw_probabilities])
        df_prediction_proba.columns = ['Retain', 'Reward', 'Nurture', 'Re-Engage']
        
        st.markdown("---")
        
        # --- Display Soft Prediction Probabilities Table ---
        st.subheader('Predicted Cluster Probabilities')
        st.dataframe(
            df_prediction_proba,
            column_config={
                # Specifying max_value=1.0 tells Streamlit to convert 0.30 directly into a beautiful 30.00% progress bar
                'Retain': st.column_config.ProgressColumn('Retain (Cluster 0)', format='%.2f%%', width='medium', min_value=0.0, max_value=1.0),
                'Reward': st.column_config.ProgressColumn('Reward (Cluster 1)', format='%.2f%%', width='medium', min_value=0.0, max_value=1.0),
                'Nurture': st.column_config.ProgressColumn('Nurture (Cluster 2)', format='%.2f%%', width='medium', min_value=0.0, max_value=1.0),
                'Re-Engage': st.column_config.ProgressColumn('Re-Engage (Cluster 3)', format='%.2f%%', width='medium', min_value=0.0, max_value=1.0),
            }, 
            hide_index=True,
            use_container_width=True
        )
        
        # --- Display Hard Prediction Outcome Banner ---
        st.subheader('Predicted Customer Segment')
        cluster_names = np.array(['Retain (Cluster 0)', 'Reward (Cluster 1)', 'Nurture (Cluster 2)', 'Re-Engage (Cluster 3)'])
        final_hard_index = int(prediction) if isinstance(prediction, np.ndarray) else int(prediction)
        st.success(f"🎯 Assigned Group: **{cluster_names[final_hard_index]}**")
        
        st.markdown("---")
        
        # --- LIVE LOCAL SHAP WATERFALL VISUALIZATION ---
        st.subheader('🧬 Feature Contribution Explanation (SHAP Waterfall)')
        st.write('This waterfall plot shows how much each input feature pushed the model toward this specific group assignment:')
        
        # Compute live SHAP values specifically for the active user input row
        # TreeExplainer outputs an array shape of (samples, features, classes) for multiclass models
        shap_values_matrix = explainer.shap_values(query_features)
        
        # Slice index 0 for the single row matrix vector, and target the active predicted class dimension
        live_feature_contributions = shap_values_matrix[0, :, final_hard_index]
        base_value = explainer.expected_value[final_hard_index]
        
        # Reconstruct the authenticated SHAP Explanation object container
        shap_explanation_live = shap.Explanation(
            values=live_feature_contributions,
            base_values=base_value,
            data=query_features.iloc,
            feature_names=query_features.columns
        )
        
        # Disable LaTeX plotting text engine to prevent layout crashes on '$' signs
        plt.rcParams['text.usetex'] = False
        
        # Draw the plot inside a Matplotlib figure object to center it beautifully at 800px width
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_explanation_live, show=False)
        plt.tight_layout()
        
        shap_col1, shap_col2, shap_col3 = st.columns([1.5, 5, 1.5])
        with shap_col2:
            st.pyplot(fig)
            
    except Exception as e:
        st.error("❌ **Prediction Engine Workspace Exception:**")
        st.warning(f"System Message: {str(e)}")

