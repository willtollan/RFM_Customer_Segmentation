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
    
    # Initialize the TreeExplainer directly on the Random Forest estimator matching your notebook
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
        prediction = loaded_model.predict(query_features)
        prediction_proba = loaded_model.predict_proba(query_features)
        
        # --- FIXED: ACCURATE MULTI-OUTPUT NESTED LIST UNPACKING ENGINE ---
        if isinstance(prediction_proba, list) and len(prediction_proba) == 4:
            extracted_probs = []
            for cluster_output in prediction_proba:
                prob_presence = float(np.asarray(cluster_output).flatten()[1])
                extracted_probs.append(prob_presence)
            raw_scores = np.array(extracted_probs)
        else:
            raw_scores = np.asarray(prediction_proba).flatten()
            
        # Normalization Step: Forces columns to add up to exactly 1.0
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
                'Retain': st.column_config.ProgressColumn('Retain (Cluster 0)', format='%.4f', min_value=0.0, max_value=1.0),
                'Reward': st.column_config.ProgressColumn('Reward (Cluster 1)', format='%.4f', min_value=0.0, max_value=1.0),
                'Nurture': st.column_config.ProgressColumn('Nurture (Cluster 2)', format='%.4f', min_value=0.0, max_value=1.0),
                'Re-Engage': st.column_config.ProgressColumn('Re-Engage (Cluster 3)', format='%.4f', min_value=0.0, max_value=1.0),
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
        
        # --- FIXED: USING EXPLICIT EXPERIMENTAL MATRIX METHOD (MATCHES PROTOTYPE) ---
        shap_values_matrix = explainer.shap_values(query_features)
        shap_array = np.asarray(shap_values_matrix)
        
        # Handle index tracking dimensional profiles safely for multi-output lists
        if shap_array.ndim == 3:
            # Slices user row 0, pulls features, grabs true presence likelihood mapping columns
            live_values = shap_array[0, :, final_hard_index]
        else:
            live_values = shap_array[0, :, final_hard_index]
            
        # Hardcode the baseline expected value to 0.25 to prevent shifting errors
        fixed_base_expected_value = 0.25
        
        # Reconstruct the exact SHAP Explanation object structure from your notebook cells
        shap_explanation_live = shap.Explanation(
            values=live_values,
            base_values=fixed_base_expected_value,
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
        col1, col2, col3 = st.columns(3)
        with col2:
            st.pyplot(fig)
            
    except Exception as e:
        st.error("❌ **Prediction Engine Live Workspace Error Exception:**")
        st.warning(f"System Message: {str(e)}")



