import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

model = pickle.load(open('model.pkl', 'rb'))

st.title('House Price Prediction Dashboard')
st.markdown('### Enter house details below to predict its price')