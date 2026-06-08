import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Classification MVP", layout="wide")
st.title('🔮 Customer Segmentation Live Inference Classifier')
st.info('This MVP builds your model from scratch to guarantee perfect alignment between the data table and the SHAP waterfall plot.')

# ----------------------------------------------------
# 1. CORE DATA LOADERS
# ----------------------------------------------------
@st.cache_data
def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_labeled_data(file_path):
    return pd.read_csv(file_path)

# --- Load Base Files (Cached) ---
try:
    df_preprocessed = load_preprocessed_data('data/preprocessed_data.csv')
    df_labeled = load_labeled_data('data/preprocessed_labelled_data.csv')
except FileNotFoundError as e:
    st.error(f"❌ Missing file error: {e}. Please ensure your datasets exist in the data/ folder.")

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
# 2. LIVE MODEL TRAINING & SHAP EXPLAINER INITIALIZATION
# ----------------------------------------------------
if 'df_labeled' in locals():
    @st.cache_resource
    def train_live_model_and_explainer(_data):
        # Isolate features and target column exactly matching your working notebook configurations
        X_train_live = _data[['MonetaryValue', 'Frequency', 'Recency']]
        y_train_live = _data['Cluster']  
        
        # Train a clean, standard single-target Multi-Class Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_live, y_train_live)
        
        # Initialize the TreeExplainer straight on the freshly trained classifier
        explainer = shap.TreeExplainer(clf)
        return clf, explainer

    with st.spinner('Training classification model live...'):
        active_model, active_explainer = train_live_model_and_explainer(df_labeled)

# ----------------------------------------------------
# 3. RUN MODEL INFERENCE
# ----------------------------------------------------
if 'active_model' in locals():
    try:
        # Enforce strict column order structure matching your notebook's X_train matrix format
        training_features = ['MonetaryValue', 'Frequency', 'Recency']
        query_features = input_df[training_features]
        
        # Execute predictions (Decimals bounded exactly between 0.0 and 1.0)
        prediction = active_model.predict(query_features)
        prediction_proba = active_model.predict_proba(query_features)
        
        # Convert probability matrix directly into a DataFrame (Sums up exactly to 1.0)
        df_prediction_proba = pd.DataFrame(prediction_proba)
        df_prediction_proba.columns = ['Retain', 'Reward', 'Nurture', 'Re-Engage']
        
        # --- Display Soft Predictions Data Table Grid ---
        st.subheader('1. Predicted Cluster Probabilities')
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
        st.subheader('2. Predicted Customer Segment')
        cluster_names = np.array(['Retain (Cluster 0)', 'Reward (Cluster 1)', 'Nurture (Cluster 2)', 'Re-Engage (Cluster 3)'])
        final_hard_index = int(prediction[0])
        st.success(f"🎯 Assigned Group: **{cluster_names[final_hard_index]}**")
        
        st.markdown("---")
        
        # ----------------------------------------------------
        # 4. LIVE SHAP WATERFALL VISUALIZATION CONTEXT
        # ----------------------------------------------------
        st.subheader('3. Live SHAP Feature Contribution Explanation')
        
        # Compute live raw SHAP matrix values for the active user input row
        # TreeExplainer outputs an array shape of (samples, features, classes) for standard multiclass models
        shap_values_matrix = active_explainer.shap_values(query_features)
        
        # Slice index 0 for the single row matrix vector, and target the active predicted class dimension column
        live_values = shap_values_matrix[0, :, final_hard_index]
        base_value = active_explainer.expected_value[final_hard_index]
        
        # Reconstruct the authenticated SHAP Explanation object container matching your notebook precisely
        shap_explanation_live = shap.Explanation(
            values=live_values,
            base_values=base_value,
            data=query_features.iloc[0],
            feature_names=query_features.columns
        )
        
        # Disable LaTeX string layout parsing warnings to prevent graph canvas generation failures
        plt.rcParams['text.usetex'] = False
        
        # Capture plot canvas inside a native Matplotlib figure framework to force elegant layout width profiles
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_explanation_live, show=False)
        plt.tight_layout()
        
        # Position chart comfortably in the center column
        col1, col2, col3 = st.columns([1.5, 5, 1.5])
        with col2:
            st.pyplot(fig)
            
    except Exception as e:
        st.error("❌ **Prediction Engine Live Workspace Error Exception:**")
        st.warning(f"System Message: {str(e)}")



