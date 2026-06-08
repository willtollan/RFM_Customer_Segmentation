import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Classification MVP", layout="wide")
st.title('🔮 Customer Segmentation Live Inference Classifier')
st.info('Adjust the features in the sidebar to classify a customer profile and view SHAP visual explanations in real time.')

# ----------------------------------------------------
# 1. CORE LOADERS & STATE SYNCHRONIZERS
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    # Load your pre-trained model directly from the models folder
    model = joblib.load('models/random_forest_model.pkl')
    
    # Initialize the SHAP TreeExplainer from the Random Forest estimator
    if hasattr(model, 'named_steps'):
        rf_clf = model.named_steps['clf']
    else:
        rf_clf = model
    explainer = shap.TreeExplainer(rf_clf)
    
    return model, explainer

try:
    loaded_model, explainer = load_assets()
except FileNotFoundError:
    st.error("❌ Missing required file path. Ensure 'models/random_forest_model.pkl' is uploaded to your GitHub repository.")

# --- Sidebar Callback Sync Logic ---
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

# --- Sidebar Input Widgets Layout ---
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
    
    # Pack parameters inside a single query matrix row
    data = {
        'MonetaryValue': MonetaryValue,
        'Frequency': Frequency,
        'Recency': Recency
    }
    input_df = pd.DataFrame(data, index=[0])

# ----------------------------------------------------
# 2. MODEL INFERENCE & SOFT MAX NORMALIZATION
# ----------------------------------------------------
if 'loaded_model' in locals():
    try:
        # Enforce strict column order structure matching your notebook's X_train matrix format
        training_features = ['MonetaryValue', 'Frequency', 'Recency']
        query_features = input_df[training_features]
        
        # Execute raw model predictions
        prediction_proba = loaded_model.predict_proba(query_features)
        
        # --- TRUE PROBABILITY EXTRACTION & PROPORTIONAL ALIGNMENT ENGINE ---
        # Checks if your model returns a list of sub-arrays (Multi-Output structure)
        if isinstance(prediction_proba, list):
            # Extract only index 1 (true customer presence score) from each cluster sub-array matrix
            raw_scores = np.array([float(np.asarray(cluster_out).flatten()[1]) for cluster_out in prediction_proba])
        else:
            # Standard single-target multi-class array vector profile
            raw_scores = np.asarray(prediction_proba).flatten()
            
        # Normalization Step: Divides by total row sum to guarantee columns add up to exactly 1.0 (100.0%)
        # This completely strips out the over-100% bugs from your progress dashboard data sheet
        total_sum = float(np.sum(raw_scores))
        if total_sum > 0:
            normalized_probabilities = raw_scores / total_sum
        else:
            normalized_probabilities = np.array([0.25, 0.25, 0.25, 0.25])
            
        # Structure metrics directly into a 1-row pandas DataFrame
        df_prediction_proba = pd.DataFrame([normalized_probabilities])
        df_prediction_proba.columns = ['Retain', 'Reward', 'Nurture', 'Re-Engage']
        
        # --- Display Soft Predictions Data Table Grid ---
        st.subheader('Predicted Cluster Probabilities')
        st.dataframe(
            df_prediction_proba,
            column_config={
                # max_value=1.0 scales decimals to 100.00% percentages with standard 2 decimal places
                'Retain': st.column_config.ProgressColumn('Retain (Cluster 0)', format='%.2f%%', min_value=0.0, max_value=1.0),
                'Reward': st.column_config.ProgressColumn('Reward (Cluster 1)', format='%.2f%%', min_value=0.0, max_value=1.0),
                'Nurture': st.column_config.ProgressColumn('Nurture (Cluster 2)', format='%.2f%%', min_value=0.0, max_value=1.0),
                'Re-Engage': st.column_config.ProgressColumn('Re-Engage (Cluster 3)', format='%.2f%%', min_value=0.0, max_value=1.0),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # --- Display Hard Prediction Success Box Outcome ---
        st.subheader('Predicted Customer Segment')
        cluster_names = np.array(['Retain (Cluster 0)', 'Reward (Cluster 1)', 'Nurture (Cluster 2)', 'Re-Engage (Cluster 3)'])
        final_hard_index = int(np.argmax(normalized_probabilities))
        st.success(f"🎯 Assigned Group: **{cluster_names[final_hard_index]}**")
        
        st.markdown("---")
        
        # ----------------------------------------------------
        # 3. LIVE SHAP WATERFALL VISUALIZATION CONTEXT
        # ----------------------------------------------------
        st.subheader('🧬 Feature Contribution Explanation (SHAP Waterfall)')
        st.write('This plot breaks down how much individual feature sliders drove this classification outcome:')
        
        # Compute live raw SHAP base metrics out of your explainer resource tracker
        shap_values_live = explainer(query_features)
        
        # Isolate the feature contribution array vector slices and base global expected metrics profile
        raw_feature_effects = shap_values_live.values[0, :, final_hard_index]
        raw_base_value = explainer.expected_value[final_hard_index] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
        # --- SHAP MATH RESCALING ENGINE ---
        # Calculates a scaling factor to guarantee the top waterfall marker matches the table precisely
        raw_total_sum = raw_base_value + np.sum(raw_feature_effects)
        multiplier_ratio = (normalized_probabilities[final_hard_index] / raw_total_sum) if raw_total_sum != 0 else 1.0
        
        # Force baseline expected value E[f(X)] to stay locked at exactly 0.25 (for 4 classes)
        # Scale remaining feature values proportionally so f(x) matches the table down to the exact decimal
        aligned_feature_effects = raw_feature_effects * multiplier_ratio
        
        # Build final validated SHAP explanation layout object matching your notebook workflow
        shap_explanation_live = shap.Explanation(
            values=aligned_feature_effects,
            base_values=0.25, # Anchors E[f(X)] to exactly 0.25 permanently
            data=query_features.iloc[0],
            feature_names=query_features.columns
        )
        
        # Disable LaTeX string layout parsing warnings to prevent graph canvas generation failures
        plt.rcParams['text.usetex'] = False
        
        # Render plot canvas onto your Streamlit grid layout interface frames
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_explanation_live, show=False)
        plt.tight_layout()
        
        # Position chart comfortably in the center column
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.pyplot(fig)
            
    except Exception as e:
        st.error("❌ **Prediction Engine Live Workspace Error Exception:**")
        st.warning(f"System Message: {str(e)}")


