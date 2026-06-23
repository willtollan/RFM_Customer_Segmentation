import streamlit as st
import numpy as np
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

# Format current input features into a clean 1-row query DataFrame matrix matching X_test schema
custom_features = {
    'MonetaryValue': monetary_input,
    'Frequency': frequency_input,
    'Recency': recency_input
}
input_df = pd.DataFrame([custom_features])[X_test.columns].astype(X_test.dtypes)

# --- RUN LIVE INFERENCE AND METRICS RENDERING ---
st.subheader('1. Live Model Inference')

# Apply model to make predictions (Decimals bounded exactly between 0.0 and 1.0)
prediction = rf_clf.predict(input_df)
prediction_proba = rf_clf.predict_proba(input_df)

# Convert probability matrix directly to a DataFrame (Sums up exactly to 1.0)
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = [cluster_labels[0], cluster_labels[1], cluster_labels[2], cluster_labels[3]]

# Render the probability data sheet using interactive Progress Columns
st.dataframe(
    df_prediction_proba,
    column_config={
        'RETAIN': st.column_config.ProgressColumn('RETAIN (Class 0)', format='%.4f', min_value=0.0, max_value=1.0),
        'REWARD': st.column_config.ProgressColumn('REWARD (Class 1)', format='%.4f', min_value=0.0, max_value=1.0),
        'NURTURE': st.column_config.ProgressColumn('NURTURE (Class 2)', format='%.4f', min_value=0.0, max_value=1.0),
        'RE-ENGAGE': st.column_config.ProgressColumn('RE-ENGAGE (Class 3)', format='%.4f', min_value=0.0, max_value=1.0),
    },
    hide_index=True,
    use_container_width=True
)

# Extract and display the hard prediction point integer value outcome
final_predicted_class = int(prediction[0])
cluster_names = np.array([cluster_labels[0], cluster_labels[1], cluster_labels[2], cluster_labels[3]])
st.success(f"🎯 Assigned Cluster Outcome: **{cluster_names[final_predicted_class]} (Class {final_predicted_class})**")

st.markdown("---")

# --- MAIN DASHBOARD INTERPRETATION LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⏱️ Live Local Explanation (Waterfall Plot)")
    st.caption(f"Visualizing feature transitions pushing this specific client toward the **{cluster_names[final_predicted_class]}** cluster.")
    
    try:
        # Compute the live SHAP matrix values for the active user input row
        # TreeExplainer outputs an array shape of (samples, features, classes) for multiclass models
        shap_values_live = explainer.shap_values(input_df)
        
        # Slice index 0 for the single row matrix vector, and target the active predicted class dimension
        live_values = shap_values_live[0, :, final_predicted_class]
        base_value = explainer.expected_value[final_predicted_class]
        
        # Reconstruct the authenticated SHAP Explanation object container
        shap_explanation_live = shap.Explanation(
            values=live_values,
            base_values=base_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        )
        
        # Disable LaTeX formatting style parsing strings to prevent layout crashes on '$' tokens
        plt.rcParams['text.usetex'] = False
        
        # Capture plot canvas inside a native Matplotlib figure framework
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 4.5))
        shap.plots.waterfall(shap_explanation_live, show=False)
        plt.title(f"Local Adjustments for Class {final_predicted_class}: {cluster_names[final_predicted_class]}", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_waterfall, clear_figure=True)
        
    except Exception as e:
        st.error(f"❌ **Local SHAP Processing Error:** {str(e)}")

with col2:
    st.subheader("🌎 Historical Macro View (Global Beeswarm Plot)")
    st.caption(f"Reviewing baseline feature weight trends for the **{cluster_names[final_predicted_class]}** cohort across the entire test set.")
    
    try:
        # Pre-calculate global test set SHAP arrays for the currently selected active cluster
        @st.cache_data
        def compute_cached_global_shap(_explainer_engine, _test_df):
            return _explainer_engine(_test_df, check_additivity=False)
            
        global_shap_values = compute_cached_global_shap(explainer, X_test)
        class_global_explanation = global_shap_values[:, :, final_predicted_class]
        
        # Handle Matplotlib figure drawing for global visualization
        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(8, 5))
        shap.plots.beeswarm(class_global_explanation, max_display=3, show=False)
        plt.title(f"Global Cohort Weight: {cluster_names[final_predicted_class]}", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_beeswarm, clear_figure=True)
        
    except Exception as e:
        st.error(f"❌ **Global SHAP Processing Error:** {str(e)}")



