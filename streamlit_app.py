import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set page configuration to wide layout
st.set_page_config(page_title="Customer Cluster Dashboard", layout="wide")

# --- CACHED DATA & MODEL LOADING ---
@st.cache_data
def load_datasets():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    return X_train, X_test

@st.cache_resource
def load_model_and_explainer(X_train):
    loaded_model = joblib.load("models/random_forest_model.pkl")
    rf_clf = loaded_model.named_steps['clf']
    
    # Clean Twin: Deep copy the classifier BEFORE passing it to SHAP.
    # This guarantees predict_proba always returns standard 0.0 - 1.0 probabilities.
    import copy
    metric_clf = copy.deepcopy(rf_clf)
    
    # Initialize the explainer (this is what modifies rf_clf's output format to log-odds/votes)
    explainer = shap.TreeExplainer(rf_clf, data=X_train)
    return metric_clf, explainer

# Load components
try:
    X_train, X_test = load_datasets()
    metric_clf, explainer = load_model_and_explainer(X_train)
except Exception as e:
    st.error(f"Error loading files. Check repository folders: {e}")
    st.stop()

# Class configuration mapping
cluster_labels = {
    0: 'RETAIN',
    1: 'REWARD',
    2: 'NURTURE',
    3: 'RE-ENGAGE'
}

# --- APPLICATION HEADER ---
st.title("🎯 Customer Segment Prediction & SHAP Interpretation")
st.markdown("Interact with customer features below to evaluate classifications and explore live structural insights.")
st.write("---")

# --- SIDEBAR FOR INTERACTIVE INPUTS ---
st.sidebar.header("🕹️ Customer Feature Inputs")

monetary_input = st.sidebar.number_input(
    "Monetary Value ($)", 
    min_value=float(X_test['MonetaryValue'].min()), 
    max_value=float(X_test['MonetaryValue'].max()), 
    value=float(X_test['MonetaryValue'].median()),
    step=10.0
)

frequency_input = st.sidebar.number_input(
    "Frequency (Visits)", 
    min_value=float(X_test['Frequency'].min()), 
    max_value=float(X_test['Frequency'].max()), 
    value=float(X_test['Frequency'].median()),
    step=1.0
)

recency_input = st.sidebar.number_input(
    "Recency (Days)", 
    min_value=float(X_test['Recency'].min()), 
    max_value=float(X_test['Recency'].max()), 
    value=float(X_test['Recency'].median()),
    step=1.0
)

# --- WORKFLOW PROCESSING ---
custom_features = {
    'MonetaryValue': monetary_input,
    'Frequency': frequency_input,
    'Recency': recency_input
}
custom_df = pd.DataFrame([custom_features])[X_test.columns].astype(X_test.dtypes)

# FIXED: We use metric_clf here to guarantee standard 0-1 probability values
probabilities = metric_clf.predict_proba(custom_df)[0]
predicted_class_int = int(metric_clf.predict(custom_df)[0])
predicted_class_name = cluster_labels[predicted_class_int]
predicted_class_prob = probabilities[predicted_class_int]

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Classification Metrics")
    
    m1, m2 = st.columns(2)
    m1.metric("Predicted Segment", f"{predicted_class_name} (Class {predicted_class_int})")
    m2.metric("Prediction Confidence", f"{predicted_class_prob * 100:.2f}%")
    
    st.write("---")
    st.subheader("⏱️ Live Local Explanation (Waterfall Plot)")
    st.caption(f"Visualizing feature transitions pushing this specific client toward the **{predicted_class_name}** cluster.")
    
    custom_shap_values = explainer(custom_df, check_additivity=False)
    predicted_explanation = custom_shap_values[0, :, predicted_class_int]
    
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(predicted_explanation, show=False)
    plt.title(f"Local Adjustments for Class {predicted_class_int}: {predicted_class_name}", fontsize=12, pad=10)
    st.pyplot(fig_waterfall, clear_figure=True)

with col2:
    st.subheader("🌎 Historical Macro View (Global Beeswarm Plot)")
    st.caption(f"Reviewing baseline feature weight trends for the **{predicted_class_name}** cohort across the entire test set.")
    
    @st.cache_data
    def compute_cached_global_shap(_explainer_engine, _test_df):
        return _explainer_engine(_test_df, check_additivity=False)
        
    global_shap_values = compute_cached_global_shap(explainer, X_test)
    class_global_explanation = global_shap_values[:, :, predicted_class_int]
    
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(8, 4.5))
    shap.plots.beeswarm(class_global_explanation, max_display=3, show=False)
    plt.title(f"Global Cohort Weight: {predicted_class_name}", fontsize=12, pad=10)
    st.pyplot(fig_beeswarm, clear_figure=True)

