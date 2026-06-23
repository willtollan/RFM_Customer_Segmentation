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
    # Reads from the "data" folder in your repository
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    return X_train, X_test

@st.cache_resource
def load_model_and_explainer(X_train):
    # Reads from the "models" folder in your repository
    loaded_model = joblib.load("models/random_forest_model.pkl")
    rf_clf = loaded_model.named_steps['clf']
    
    # Initialize the explainer using background training data for empirical expected values
    explainer = shap.TreeExplainer(rf_clf, data=X_train)
    return rf_clf, explainer

# Load components
try:
    X_train, X_test = load_datasets()
    rf_clf, explainer = load_model_and_explainer(X_train)
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

# Dynamically set boundaries based on your test set distribution
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
# 1. Format user metrics to perfectly align with training schema structure & data types
custom_features = {
    'MonetaryValue': monetary_input,
    'Frequency': frequency_input,
    'Recency': recency_input
}
custom_df = pd.DataFrame([custom_features])[X_test.columns].astype(X_test.dtypes)

# 2. INTEGRATED PROBABILITY LOGIC: Extracts true probabilities matching Jupyter precisely
predicted_class_int = int(rf_clf.predict(custom_df)[0])
predicted_class_name = cluster_labels[predicted_class_int]

soft_prediction_probabilities = rf_clf.predict_proba(custom_df)[0]
predicted_class_prob = soft_prediction_probabilities[predicted_class_int]

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Classification Metrics")
    
    # Display clear metric callout windows
    m1, m2 = st.columns(2)
    m1.metric("Predicted Segment", f"{predicted_class_name} (Class {predicted_class_int})")
    m2.metric("Prediction Confidence", f"{predicted_class_prob * 100:.2f}%")
    
    st.write("---")
    st.subheader("⏱️ Live Local Explanation (Waterfall Plot)")
    st.caption(f"Visualizing feature transitions pushing this specific client toward the **{predicted_class_name}** cluster.")
    
    # Generate on-the-fly live local explanation using the reference script's constructor pattern
    shap_output = explainer(custom_df, check_additivity=False)
    
    # Safely unpack and assemble the structural explanation object container
    predicted_explanation = shap.Explanation(
        values=shap_output.values[0, :, predicted_class_int],
        base_values=shap_output.base_values[0, predicted_class_int],
        data=custom_df.iloc[0],
        feature_names=custom_df.columns
    )
    
    # Handle Matplotlib figure drawing to safely render in Streamlit
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(predicted_explanation, show=False)
    plt.title(f"Local Adjustments for Class {predicted_class_int}: {predicted_class_name}", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig_waterfall, clear_figure=True)

with col2:
    st.subheader("🌎 Historical Macro View (Global Beeswarm Plot)")
    st.caption(f"Reviewing baseline feature weight trends for the **{predicted_class_name}** cohort across the entire test set.")
    
    # Pre-calculate global test set SHAP arrays for the currently selected active cluster
    @st.cache_data
    def compute_cached_global_shap(_explainer_engine, _test_df):
        return _explainer_engine(_test_df, check_additivity=False)
        
    global_shap_values = compute_cached_global_shap(explainer, X_test)
    class_global_explanation = global_shap_values[:, :, predicted_class_int]
    
    # Handle Matplotlib figure drawing for global visualization
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(8, 4.5))
    shap.plots.beeswarm(class_global_explanation, max_display=3, show=False)
    plt.title(f"Global Cohort Weight: {predicted_class_name}", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig_beeswarm, clear_figure=True)

