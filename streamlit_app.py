import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Customer Cluster Explainer", layout="wide")
st.title("🛍️ Customer Cluster Predictor & SHAP Explainer")

LABELS = {
    0: 'RETAIN',
    1: 'REWARD',
    2: 'NURTURE',
    3: 'RE-ENGAGE'
}

# 2. Cached Training Pipeline (Runs once on app startup)
@st.cache_resource
def train_and_initialize_explainer():
    # Load raw data from your data/ folder
    df = pd.read_csv("data/preprocessed_labelled_data.csv")
    
    X = df[['MonetaryValue', 'Frequency', 'Recency']]
    y = df['Cluster']
    
    # Train/Test Split (exactly as configured in your original training script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )
    
    # Train the Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # FIX: Explicitly pass X_train here to calculate empirical data distribution means.
    # This prevents the baseline from defaulting to an unweighted 0.25.
    explainer = shap.TreeExplainer(rf_clf, data=X_train)
    
    return rf_clf, explainer, X_test

# Trigger training or pull from cache
with st.spinner("🔄 Training Random Forest and initializing SHAP explainer..."):
    try:
        rf_clf, explainer, X_test = train_and_initialize_explainer()
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please upload 'preprocessed_labelled_data.csv' into your 'data/' folder on GitHub.")
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


