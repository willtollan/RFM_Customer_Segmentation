import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris  # Offline dataset import

st.set_page_config(page_title="SHAP Multi-Class Prototype", layout="wide")
st.title('🌸 SHAP Multi-Class Classification Prototype')
st.info('This minimal app trains a Random Forest Classifier on the Iris dataset completely offline to test your SHAP visualizations.')

# ----------------------------------------------------
# 1. LOAD AND PREPARE DATA (OFFLINE)
# ----------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Load iris dataset directly from scikit-learn memory
    iris = load_iris()
    
    # Structure features into a clean DataFrame
    X = pd.DataFrame(iris.data, columns=['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm'])
    
    # Target is already a flat 1D Series of class integers (0, 1, 2)
    y = pd.Series(iris.target)
    
    return X, y

X_raw, y_raw = load_and_prepare_data()

# ----------------------------------------------------
# 2. TRAIN THE CLASSIFIER AND CACHE WORKSPACE ARTIFACTS
# ----------------------------------------------------
@st.cache_resource
def train_model_and_explainer(X, y):
    # Train a clean, standard single-target Multi-Class Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Initialize the TreeExplainer directly on the trained model object
    explainer = shap.TreeExplainer(clf)
    return clf, explainer

model, explainer = train_model_and_explainer(X_raw, y_raw)

# ----------------------------------------------------
# 3. INTERACTIVE SIDEBAR CONTROLS (USER FEATURES INPUTS)
# ----------------------------------------------------
with st.sidebar:
    st.header('Adjust Input Features')
    sepal_len = st.slider('Sepal Length (cm)', float(X_raw['sepal_length_cm'].min()), float(X_raw['sepal_length_cm'].max()), float(X_raw['sepal_length_cm'].median()))
    sepal_wid = st.slider('Sepal Width (cm)', float(X_raw['sepal_width_cm'].min()), float(X_raw['sepal_width_cm'].max()), float(X_raw['sepal_width_cm'].median()))
    petal_len = st.slider('Petal Length (cm)', float(X_raw['petal_length_cm'].min()), float(X_raw['petal_length_cm'].max()), float(X_raw['petal_length_cm'].median()))
    petal_wid = st.slider('Petal Width (cm)', float(X_raw['petal_width_cm'].min()), float(X_raw['petal_width_cm'].max()), float(X_raw['petal_width_cm'].median()))
    
    # Format current slider features rows into a clean 1-row query DataFrame matrix
    data = {
        'sepal_length_cm': sepal_len,
        'sepal_width_cm': sepal_wid,
        'petal_length_cm': petal_len,
        'petal_width_cm': petal_wid
    }
    input_df = pd.DataFrame(data, index=[0])

# ----------------------------------------------------
# 4. RUN LIVE INFERENCE AND DATA TABLE RENDERING
# ----------------------------------------------------
st.subheader('1. Live Model Inference')

# Apply model to make predictions (Decimals bounded exactly between 0.0 and 1.0)
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Convert probability matrix directly to a DataFrame (Sums up exactly to 1.0)
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Setosa', 'Versicolor', 'Virginica']

# Render the probability data sheet using interactive Progress Columns
st.dataframe(
    df_prediction_proba,
    column_config={
        'Setosa': st.column_config.ProgressColumn('Setosa (Class 0)', format='%.4f', min_value=0.0, max_value=1.0),
        'Versicolor': st.column_config.ProgressColumn('Versicolor (Class 1)', format='%.4f', min_value=0.0, max_value=1.0),
        'Virginica': st.column_config.ProgressColumn('Virginica (Class 2)', format='%.4f', min_value=0.0, max_value=1.0),
    },
    hide_index=True,
    use_container_width=True
)

# Extract and display the hard prediction point integer value outcome
final_predicted_class = int(prediction[0])
species_names = np.array(['Setosa', 'Versicolor', 'Virginica'])
st.success(f"🎯 Assigned Species Outcome: **{species_names[final_predicted_class]}**")

st.markdown("---")

# ----------------------------------------------------
# 5. GENERATE LIVE SHAP WATERFALL VISUALIZATION
# ----------------------------------------------------
st.subheader('2. Live SHAP Feature Contribution Explanation')

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
    
    # Capture plot canvas inside a native Matplotlib figure framework to force elegant layout width profiles
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_explanation_live, show=False)
    plt.tight_layout()
    
    # Center the graphic beautifully using empty column padding blocks
    col1, col2, col3 = st.columns([1.5, 5, 1.5])
    with col2:
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ **SHAP Processing Error:** {str(e)}")



