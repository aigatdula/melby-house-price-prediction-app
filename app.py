import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pycaret.regression import *

# Data preparation 23
@st.cache()
def load_data1():
    lat_long = pd.read_csv('Melbourne_housing_FULL.csv', usecols=['latitude','longitude'])
    lat_long = lat_long.dropna()
    return lat_long
    
@st.cache()
def load_data2():
    data = pd.read_csv('Melbourne_housing_FULL.csv', usecols=['Rooms', 'Price', 'Distance', 'Bathroom', 'Car','Landsize'])
    return data


def main():

    st.title("Welcome to Melbourne's House Price Prediction App")

    check_data = st.checkbox("Show me the Melbourne Map and House Listings")
    if check_data:
        lat_long = load_data1()
        st.map(lat_long)
    
    #input the numbers
    data = load_data2()
    sqft_liv = st.slider("What is the landsize in meters?",150,1000,value=500,step=50 )
    bath     = st.slider("How many bathrooms?",1,3,value=1)
    rooms      = st.slider("How many bedrooms?",int(data.Rooms.min()),6,value=2 )
    Car    = st.slider("How many car-port do you want?",int(data.Car.min()),2,value=1 )   
    Distance = st.slider("Distance from CBD in km?",int(data.Distance.min()),40,value=5 )
#load pickle
    saved_final_lightgbm = load_model('Light Gradient Boosting Machine Model 01Jun2020')
    
# Create the pandas DataFrame 
    data = [[sqft_liv, bath,rooms,Car,Distance]] 
    data_unseen = pd.DataFrame(data, columns = ['Landsize', 'Bathroom','Rooms','Car','Distance']) 
    pred = predict_model(saved_final_lightgbm, data=data_unseen)
    pred = pred.rename(columns={"Label": "Predicted Price"}, errors="raise")
    format_mapping={'Predicted Price': '${:,.0f}'}
    for key, value in format_mapping.items():
        pred[key] = pred[key].apply(value.format)
    st.dataframe(pred)

    st.sidebar.info('Melbourne House Price Prediction app by: aigatdula4@yahoo.com')

if __name__ == '__main__':
    main()