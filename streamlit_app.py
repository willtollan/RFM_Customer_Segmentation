import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

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

# 4. Generate Predictions & Compute SHAP Values Upfront
hard_prediction = rf_clf.predict(user_input_df)[0]
predicted_label = LABELS[hard_prediction]

# Compute SHAP values using the modern __call__ syntax for the user's specific inputs
shap_output = explainer(user_input_df)

# Extract and sum SHAP components to calculate the exact f(x) probability for the predicted class
base_value = shap_output.base_values[0, hard_prediction]
shap_values_sum = shap_output.values[0, :, hard_prediction].sum()
fx_probability = float(base_value + shap_values_sum)

# 5. Display Predictions Dashboard
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("🎯 Assignment")
    st.metric(label="Predicted Cluster", value=predicted_label)

with col_m2:
    st.subheader("📊 Model Confidence")
    # Displays the exact f(x) probability directly as a metric callout card
    st.metric(label="Prediction Probability f(x)", value=f"{fx_probability * 100:.2f}%")

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



