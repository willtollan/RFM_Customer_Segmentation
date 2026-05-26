import streamlit as st
import pandas as pd

st.title('🎈 RFM Customer Segmentation')

st.info('This app builds a machine learning model')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df
