import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

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
def load_rf_best_params(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_rf_metrics(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_rf_report(file_path):
    return pd.read_csv(file_path)

# --- Safely Initialize Base Processing Dependencies (Cached) ---
try:
    df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
    df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
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
    
    # --- Boxplot Plots Section (Standard Format Plot) ---
    st.subheader('Cluster Boxplot Plots by Feature')
    st.write('Distribution spread and density of Recency, Frequency, and Monetary Value across each cluster:')
    
    v_col1, v_col2, v_col3 = st.columns([1.5, 5, 1.5])
    with v_col2:
        try:
            st.image('images/cluster_boxplot_by_features.png', width=900)
        except FileNotFoundError:
            st.error("Could not find 'images/cluster_boxplot_by_features.png'.")

# ----------------------------------------------------
# 4. RANDOM FOREST CLASSIFIER PERFORMANCE METRICS
# ----------------------------------------------------

with st.expander('Surrogate Classifier', expanded=False):
    
    # --- Model Selection ---
    st.subheader('Cross-Validation Results across Multiple Classifiers')
    st.write('Random Forest Classifier achieved the strongest cross-validation performance')
    
    param_col1, param_col2, param_col3 = st.columns([1.5, 5, 1.5])
    with param_col2:
        try:
            st.image('images/classifier_performance_comparison.png', width=900)
        except FileNotFoundError:
            st.error("Could not find 'images/classifier_performance_comparison.png'.")
            
    st.markdown("---")
    
    # --- Random Forest Best Parameters ---
    st.subheader('Random Forest Best Parameters')
    st.write('The optimal hyperparameters found during the grid search tuning optimization phase (Randomized Cross-Validation Search):')
    
    met_col1, met_col2, met_col3 = st.columns([1.5, 6, 1.5])
    with met_col2:
        try:
            df_rf_best_params = load_rf_best_params('data/RF_best_params.csv')
            st.dataframe(df_rf_best_params, use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.error("Could not find 'data/RF_best_params.csv'.")
        
    st.markdown("---")
    
    # --- Key Metrics Section ---
    st.subheader('Key Metrics')
    st.write('Overall evaluation metrics for the tuned Random Forest classification model:')
    
    met_col1, met_col2, met_col3 = st.columns([1.5, 2, 1.5])
    with met_col2:
        try:
            df_rf_metrics = load_rf_metrics('data/tuned_RF_key_metrics.csv')
            st.dataframe(df_rf_metrics, use_container_width=True, hide_index=True)
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
            df_rf_report.columns.values[0] = "Labels"
            st.dataframe(df_rf_report, use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.error("Could not find 'data/tuned_RF_classification_report.csv'.")
        
    st.markdown("---")
    
    # --- Confusion Matrix Section (Small Format Plot) ---
    st.subheader('Confusion Matrix')
    st.write('Matrix visualising the actual versus predicted classification distributions on test data subsets:')
    
    cm_col1, cm_col2, cm_col3 = st.columns([1, 2, 1])
    with cm_col2:
        st.image('images/tuned_RF_confusion_matrix.png', use_column_width=True)

    # --- Random Forest Feature Importances ---
    st.subheader('Random Forest Feature Importances')
    st.write('Visualization of how much each feature contributes to the predictive power of the Random Forest model using Mean Decrease in Impurity (MDI) / Gini Importance')
    
    cm_col1, cm_col2, cm_col3 = st.columns([1, 2, 1])
    with cm_col2:
        st.image('images/RF_feature_importances.png', use_column_width=True)


# ----------------------------------------------------
# Classification Prediction and SHAP Explainability
# ----------------------------------------------------

# 1. Page Configuration
st.set_page_config(page_title="Customer Cluster Explainer", layout="wide")
st.title("🛍️ Customer Cluster Predictor & SHAP Explainer")

LABELS = {
    0: 'RETAIN',
    1: 'REWARD',
    2: 'NURTURE',
    3: 'RE-ENGAGE'
}

# 2. Cached Data & Pretrained Model Loading
@st.cache_data
def load_datasets():
    # Reads from the "data" folder in your repository
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    return X_train, X_test

@st.cache_resource
def load_model_and_explainer(X_train):
    # Reads your custom pretrained model from the "models" folder
    loaded_model = joblib.load("models/random_forest_model.pkl")
    rf_clf = loaded_model.named_steps['clf']
    
    # Initialize explainer using background training data for empirical expected values
    explainer = shap.TreeExplainer(rf_clf, data=X_train)
    return rf_clf, explainer

# Load production components
try:
    X_train, X_test = load_datasets()
    rf_clf, explainer = load_model_and_explainer(X_train)
except Exception as e:
    st.error(f"⚠️ Error loading production files. Check repository paths: {e}")
    st.stop()

# --- OPTIMIZED CACHED GLOBAL SHAP ENGINE ---
@st.cache_data
def compute_cached_global_shap(_explainer_engine, _test_df):
    return _explainer_engine(_test_df, check_additivity=False)

# Pre-calculate full reference matrix for the macro-level view
global_shap_values = compute_cached_global_shap(explainer, X_test)

# 3. Sidebar Input Elements for Features
st.sidebar.header("📥 Input Customer Features")

monetary_value = st.sidebar.number_input(
    "Monetary Value ($)", 
    min_value=0.0, 
    max_value=50000.0, 
    value=150.0, 
    step=10.0
)
frequency = st.sidebar.slider(
    "Frequency (Total Visits)", 
    min_value=1, 
    max_value=100, 
    value=1, 
    step=1
)
recency = st.sidebar.slider(
    "Recency (Days Since Last Purchase)", 
    min_value=0, 
    max_value=365, 
    value=110, 
    step=1
)

# Convert inputs into a single-row DataFrame matching the model features
user_input_df = pd.DataFrame([{
    'MonetaryValue': monetary_value,
    'Frequency': frequency,
    'Recency': recency
}])
user_input_df = user_input_df[X_test.columns].astype(X_test.dtypes)

# 4. Generate SHAP Values Upfront
shap_output = explainer(user_input_df, check_additivity=False)

# Calculate the exact f(x) probability for all 4 classes using SHAP margin outputs
all_classes_probabilities = []
for class_idx in sorted(LABELS.keys()):
    class_base_value = shap_output.base_values[0, class_idx]
    class_shap_sum = shap_output.values[0, :, class_idx].sum()
    class_fx_prob = float(class_base_value + class_shap_sum)
    all_classes_probabilities.append(class_fx_prob)

# FIX: Define the prediction dynamically based on the highest probability in the f(x) space
hard_prediction = int(np.argmax(all_classes_probabilities))
predicted_label = LABELS[hard_prediction]

# Set the active prediction base value for the waterfall plot
base_value = shap_output.base_values[0, hard_prediction]

# Create a clean data presentation frame using the SHAP-derived f(x) probabilities
prob_df = pd.DataFrame({
    "Class ID": list(LABELS.keys()),
    "Cluster Cohort": list(LABELS.values()),
    "Probability f(x)": all_classes_probabilities
})

# Define a custom style function to highlight the active prediction row
def highlight_predicted_row(row):
    if row["Class ID"] == hard_prediction:
        return ['background-color: #1e4620; color: #a3e635; font-weight: bold;'] * len(row)
    return [''] * len(row)

# Apply formatting styles (Percentage display + conditional row highlight)
styled_prob_df = (
    prob_df.style
    .format({"Probability f(x)": "{:.2%}"})
    .apply(highlight_predicted_row, axis=1)
)

# 5. Display Predictions Dashboard
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("🎯 Primary Assignment")
    st.write("")  # Visual spacer
    st.metric(label="Assigned Cluster Cohort", value=predicted_label)
    st.caption(f"The input values map this customer profile directly to **Cluster {hard_prediction}**.")

with col_m2:
    st.subheader("📊 Full Cohort Probability Breakdown")
    # Render the styled probability matrix table directly
    st.dataframe(
        styled_prob_df,
        hide_index=True,
        use_container_width=True
    )

st.write("---")

# 6. Display Dual SHAP Plots (Waterfall on left, Beeswarm on right)
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("⏱️ Live Local Explanation (Waterfall Plot)")
    st.caption(f"Visualizing feature transitions pushing this specific client toward the **{predicted_label}** cluster.")
    
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 4.5))
    
    # Plot the waterfall diagram using correct structural array slicing from shap_output.
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_output.values[0, :, hard_prediction],
            base_values=base_value, 
            data=user_input_df.iloc[0],
            feature_names=user_input_df.columns
        ),
        show=False
    )
    plt.title(f"Local Adjustments for Class {hard_prediction}: {predicted_label}", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig_waterfall, clear_figure=True)

with col_plot2:
    st.subheader("🌎 Historical Macro View (Global Beeswarm Plot)")
    st.caption(f"Reviewing baseline feature weight trends for the **{predicted_label}** cohort across the entire test set.")
    
    # Extract the pre-calculated 2D slice for the currently active predicted class segment
    class_global_explanation = global_shap_values[:, :, hard_prediction]
    
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(8, 4.5))
    shap.plots.beeswarm(class_global_explanation, max_display=3, show=False)
    plt.title(f"Global Cohort Weight: {predicted_label}", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig_beeswarm, clear_figure=True)
    

# ----------------------------------------------------
# Cluster Description & Recommended Strategy
# ----------------------------------------------------

# 7. Dynamic Cluster Description & Strategy Section
st.write("---")
st.subheader("📝 Cluster Description & Recommended Strategy")

# Define descriptions, strategies, and emojis for each cluster
CLUSTER_INFO = {
    "RETAIN": {
        "emoji": "🔒",
        "description": "Customers who are moderately active and valuable, but not top-tier. They are steady but could drift away if ignored.",
        "strategy": "Keep them engaged with loyalty points, personalized recommendations, and consistent communication."
    },
    "REWARD": {
        "emoji": "🎁",
        "description": "Customers who buy frequently, spend a lot, and purchased recently. These are your best customers — loyal and high-value.",
        "strategy": "Reward them with exclusive offers, VIP programs, or early access."
    },
    "NURTURE": {
        "emoji": "🌱",
        "description": "New or low-value customers who purchased recently but haven’t yet shown loyalty or high spend. They’re at the beginning of their journey.",
        "strategy": "Nurture them with onboarding, education, and incentives to build habits."
    },
    "RE-ENGAGE": {
        "emoji": "🔄",
        "description": "Customers who haven’t purchased in a long time, spend little, and rarely buy. They are at risk of churn or already inactive.",
        "strategy": "Win them back with reactivation campaigns, discounts, or reminders."
    }
}

# Dynamically display based on prediction
cluster_info = CLUSTER_INFO.get(predicted_label, {})
emoji = cluster_info.get("emoji", "")
st.markdown(f"**Cluster Classification:** {emoji} {predicted_label}")
st.write(cluster_info.get("description", "No description available."))
st.info(f"💡 Recommended Strategy: {cluster_info.get('strategy', 'No strategy available.')}")

