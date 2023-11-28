# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


st.set_page_config(page_title='cancer_prediction',layout='wide')
st.title("Breast Cancer Prediction") 
left_column,right_column=st.columns(2)
with left_column:
    radius_mean=st.number_input(label="radius_mean")
    texture_mean=st.number_input(label="texture_mean")
    perimeter_mean=st.number_input(label="perimeter_mean")
    area_mean=st.number_input(label="area_mean")
    smoothness_mean=st.number_input(label="smoothness_mean")
    compactness_mean=st.number_input(label="compactness_mean")
    concavity_mean=st.number_input(label="concavity_mean")
    concave_points_mean=st.number_input(label="concave points_mean")
    symmetry_mean=st.number_input(label="symmetry_mean")
    fractal_dimension_mean=st.number_input(label="fractal_dimension_mean")
    radius_se=st.number_input(label="radius_se")
    texture_se=st.number_input(label="texture_se")
    perimeter_se=st.number_input(label="perimeter_se")
    area_se=st.number_input(label="area_se")
    smoothness_se=st.number_input(label="smoothness_se")
with right_column:
    compactness_se=st.number_input(label="compactness_se")
    concavity_se=st.number_input(label="concavity_se")
    concave_points_se=st.number_input(label="concave points_se")
    symmetry_se=st.number_input(label="symmetry_se")
    fractal_dimension_se=st.number_input(label="fractal_dimension_se")
    radius_worst=st.number_input(label="radius_worst")
    texture_worst=st.number_input(label="texture_worst")
    perimeter_worst=st.number_input(label="perimeter_worst")
    area_worst=st.number_input(label="area_worst")
    smoothness_worst=st.number_input(label="smoothness_worst")
    compactness_worst=st.number_input(label="compactness_worst")
    concavity_worst=st.number_input(label="concavity_worst")
    concave_points_worst=st.number_input(label="concave points_worst")
    symmetry_worst=st.number_input(label="symmetry_worst")
    fractal_dimension_worst=st.number_input(label="fractal_dimension_worst") 
if st.button("SUBMIT"):
    values=[[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,
            concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,
            area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,
            fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,
            compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]]

    patient_df=pd.DataFrame(values,columns=['radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst'])

    st.dataframe(patient_df) 
    import pickle
    with open(r"C:/Users/dhiyanesh/Downloads/cancer_prediction/cmodel.pkl",'rb') as file:
        rfc_model=pickle.load(file)
        result=rfc_model.predict(patient_df)
    if result==1:
        st.write("**The Cell is MALIGNANT**")
    else:
        st.write("**The Cell is BENIGN**")
st.write(f'<h6 style="color:rgb(0,153,153,0.35);">App Created by Lokesh</h6>',unsafe_allow_html=True)
    
