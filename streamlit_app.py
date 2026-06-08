import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Classification MVP", layout="wide")
st.title('🔮 Customer Segmentation Live Inference Classifier')

# ----------------------------------------------------
# 1. LOAD DATA & EXTRACT FEATURES IN CORRECT ORDER
# ----------------------------------------------------
@st.cache_data
def load_labeled_data(file_path):
    return pd.read_csv(file_path)

try:
    df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
except FileNotFoundError:
    st.error("❌ Missing required data file. Please ensure 'data/preprocessed_labelled_data.csv' exists.")

# Define the exact feature column order used throughout the workflow
training_features = ['MonetaryValue', 'Frequency', 'Recency']

# ----------------------------------------------------
# 2. TRAIN THE CLASSIFIER LIVE (Cached for performance)
# ----------------------------------------------------
@st.cache_resource
def train_model_and_explainer(_data, features):
    X_train_live = _data[features]
    y_train_live = _data['Cluster']  # Target column with integer tokens 0, 1, 2, 3
    
    # Train a standard multi-class Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_live, y_train_live)
    
    # Initialize TreeExplainer directly on the trained model
    explainer = shap.TreeExplainer(clf)
    return clf, explainer

if 'df_labeled' in locals():
    active_model, active_explainer = train_model_and_explainer(df_labeled, training_features)

# ----------------------------------------------------
# 3. INTERACTIVE SIDEBAR INPUT CONTROLS (State Sync)
# ----------------------------------------------------
def sync_mv_slider(): st.session_state.mv_num = st.session_state.mv_slide
def sync_mv_num(): st.session_state.mv_slide = st.session_state.mv_num
def sync_freq_slider(): st.session_state.freq_num = st.session_state.freq_slide
def sync_freq_num(): st.session_state.freq_slide = st.session_state.freq_num
def sync_rec_slider(): st.session_state.rec_num = st.session_state.rec_slide
def sync_rec_num(): st.session_state.rec_slide = st.session_state.rec_num

with st.sidebar:
    st.header('Input features')
    st.number_input('MonetaryValue Input', 5, 4000, 1634, key='mv_num', on_change=sync_mv_num)
    MonetaryValue = st.slider('MonetaryValue', 5, 4000, key='mv_slide', on_change=sync_mv_slider)
    st.markdown("---")
    st.number_input('Frequency Input', 1, 12, 1, key='freq_num', on_change=sync_freq_num)
    Frequency = st.slider('Frequency', 1, 12, key='freq_slide', on_change=sync_freq_slider)
    st.markdown("---")
    st.number_input('Recency Input', 0, 375, 113, key='rec_num', on_change=sync_rec_num)
    Recency = st.slider('Recency', 0, 375, key='rec_slide', on_change=sync_rec_slider)
    
    # Create the 1-row input DataFrame matching the precise feature order
    input_df = pd.DataFrame([{
        'MonetaryValue': MonetaryValue,
        'Frequency': Frequency,
        'Recency': Recency
    }])[training_features]

# ----------------------------------------------------
# 4. SHOW HARD AND SOFT PREDICTIONS
# ----------------------------------------------------
if 'active_model' in locals():
    try:
        # Run real-time model inference
        prediction = active_model.predict(input_df)
        prediction_proba = active_model.predict_proba(input_df)
        
        # Flatten probabilities to a clean 1D array (guarantees they sum to 1.0)
        raw_probabilities = np.asarray(prediction_proba).flatten()
        
        # Display Soft Predictions
        st.subheader('Predicted Cluster Probabilities')
        df_prediction_proba = pd.DataFrame([raw_probabilities], columns=['Retain', 'Reward', 'Nurture', 'Re-Engage'])
        st.dataframe(
            df_prediction_proba,
            column_config={
                'Retain': st.column_config.ProgressColumn('Retain (Cluster 0)', format='%.4f', min_value=0.0, max_value=1.0),
                'Reward': st.column_config.ProgressColumn('Reward (Cluster 1)', format='%.4f', min_value=0.0, max_value=1.0),
                'Nurture': st.column_config.ProgressColumn('Nurture (Cluster 2)', format='%.4f', min_value=0.0, max_value=1.0),
                'Re-Engage': st.column_config.ProgressColumn('Re-Engage (Cluster 3)', format='%.4f', min_value=0.0, max_value=1.0),
            },
            hide_index=True, use_container_width=True
        )
        
        # Display Hard Prediction
        st.subheader('Predicted Customer Segment')
        cluster_names = np.array(['Retain (Cluster 0)', 'Reward (Cluster 1)', 'Nurture (Cluster 2)', 'Re-Engage (Cluster 3)'])
        final_hard_index = int(prediction[0])
        st.success(f"🎯 Assigned Group (Hard Prediction): **{cluster_names[final_hard_index]}**")
        
        st.markdown("---")

        # ----------------------------------------------------
        # 5. GENERATE LIVE SHAP WATERFALL PLOT
        # ----------------------------------------------------
        st.subheader('🧬 Feature Contribution Explanation (SHAP Waterfall)')
        
        # Compute multi-class 3D SHAP matrix for the current user inputs
        shap_values_matrix = active_explainer.shap_values(input_df)
        
        # Slice user row 0, all features, and target the specific predicted class column
        live_feature_contributions = shap_values_matrix[0, :, final_hard_index]
        
        # Hardcode the global prior probability base expected value to exactly 0.25
        fixed_base_expected_value = 0.25
        
        # Reconstruct the authentic SHAP Explanation container
        shap_explanation_live = shap.Explanation(
            values=live_feature_contributions,
            base_values=fixed_base_expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        )
        
        # Disable LaTeX parsing engine to avoid crashes on mathematical plot symbols
        plt.rcParams['text.usetex'] = False
        
        # Render plot canvas onto the Matplotlib structure
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_explanation_live, show=False)
        plt.tight_layout()
        
        # Center the chart layout beautifully
        col1, col2, col3 = st.columns([1.5, 5, 1.5])
        with col2:
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"❌ Execution Error: {str(e)}")




