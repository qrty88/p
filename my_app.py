#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

st.sidebar.title('Car Price Prediction')


# To take feature inputs
make_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Espace'))
km=st.sidebar.slider("What is the km of your car", 0,320000, step=500)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)

# To load machine learning model
final_model = pickle.load(open('rf_model', 'rb'))
final_model_transformer = pickle.load(open('transformer', 'rb'))

# Create a dataframe using feature inputs
my_dict = {
    "make_model": make_model,
    "km": km,
    "Gearing_Type":gearing_type,
    "age": age,
    "hp_kW": hp,
}

df = pd.DataFrame.from_dict([my_dict])

st.header(" The features of ur car is below:")
st.table(df)

df2 = final_model_transformer.transform(df)

st.subheader("click predict")

if st.button("Predict"):
    prediction = final_model.predict(df2)




