import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(page_title="Customer Cluster Explainer", layout="centered")
st.title("🛍️ Customer Cluster Predictor & SHAP Explainer")

LABELS = {
    0: 'RETAIN',
    1: 'REWARD',
    2: 'NURTURE',
    3: 'RE-ENGAGE'
}

# 2. Cached Pipeline: Load pre-trained model & anchor SHAP with background data
@st.cache_resource
def load_production_assets():
    # Load your existing pre-trained model from the 'models/' folder
    with open("models/random_forest_model.pkl", "rb") as f:
        rf_clf = pickle.load(f)
        
    # Load raw data to extract the exact background split for SHAP
    df = pd.read_csv("data/preprocessed_labelled_data.csv")
    X = df[['MonetaryValue', 'Frequency', 'Recency']]
    y = df['Cluster']
    
    # Recreate the train split to grab the background data distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )
    
    # Initialize the SHAP explainer anchored on X_train to preserve true class imbalance
    explainer = shap.TreeExplainer(rf_clf, data=X_train)
    
    return rf_clf, explainer

# Trigger loading sequences or gracefully stop on error
with st.spinner("🔄 Loading pre-trained model and syncing SHAP baseline distributions..."):
    try:
        rf_clf, explainer = load_production_assets()
    except FileNotFoundError as e:
        st.error(f"⚠️ Configuration error: Verify that your dataset exists in 'data/' and your pre-trained model file exists in 'models/'. Missing: {e.filename}")
        st.stop()

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

# Convert active user configuration settings into a operational single-row Dataframe
user_input_df = pd.DataFrame([{
    'MonetaryValue': monetary_value,
    'Frequency': frequency,
    'Recency': recency
}])

# 4. Generate Predictions
hard_prediction = rf_clf.predict(user_input_df)[0]
predicted_label = LABELS[hard_prediction]

soft_prediction_probabilities = rf_clf.predict_proba(user_input_df)[0]

# 5. Display Predictions Dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Assignment")
    st.metric(label="Predicted Cluster", value=predicted_label)

with col2:
    st.subheader("📊 Probability Breakdown")
    prob_df = pd.DataFrame({
        "Cluster": list(LABELS.values()),
        "Probability": soft_prediction_probabilities
    })
    st.dataframe(
        prob_df.style.format({"Probability": "{:.2%}"}),
        hide_index=True
    )

# 6. Compute and Display SHAP Waterfall Chart
st.write("---")
st.subheader(f"🔍 SHAP Waterfall Explanation for Cluster: {predicted_label}")

# Execute calculations against current user coordinates
shap_output = explainer(user_input_df)

fig, ax = plt.subplots(figsize=(8, 4.5))

# Slices out exact classification weights mapping back to actual data base values
shap.plots.waterfall(
    shap.Explanation(
        values=shap_output.values[0, :, hard_prediction],
        base_values=shap_output.base_values[0, hard_prediction], 
        data=user_input_df.iloc[0],
        feature_names=user_input_df.columns
    ),
    show=False
)

plt.tight_layout()
st.pyplot(fig)

