#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pickle  # optional, if you want to save/load model
from xgboost import XGBRegressor
import joblib

model = joblib.load('xgb_model.pkl')

# seller_type
seller_type_map = {
    "AGENT": 15778,
    "OWNER": 5668,
    "BUILDER": 329
}

# layout_type
layout_type_map = {
    "BHK": 20791,
    "RK": 984
}

# property_type
property_type_map = {
    "Apartment": 19391,
    "Studio Apartment": 984,
    "Independent House": 824,
    "Independent Floor": 306,
    "Villa": 259,
    "Penthouse": 11
}

# locality (partial, only showing Koregaon Park)
locality_map = {"Wagholi": 2165, "Hinjewadi": 1442, "Wakad": 1253, "Kharadi": 1218, "Hadapsar": 1109, "Bavdhan": 784, "Baner": 735, "Pimple Saudagar": 653, "Wadgaon Sheri": 558, "Viman Nagar": 552, "Kothrud": 534, "Dhanori": 494, "Kondhwa": 466, "Mundhwa": 419, "Lohegaon": 371, "Dhayari": 370, "Chinchwad": 368, "Kalyani Nagar": 356, "Balewadi": 339, "Undri": 335, "Koregaon Park": 312, "Wanowrie": 272, "Ravet": 271, "Aundh": 254, "NIBM Annex Mohammadwadi": 241, "Rahatani": 220, "Akurdi": 217, "Pimple Gurav": 198, "Yerawada": 182, "Bibwewadi": 168, "Vadgaon Budruk": 158, "Pimple Nilakh": 151, "Vishrantwadi": 150, "Fursungi": 147, "Warje": 141, "Thergaon": 137, "Karve Nagar": 135, "Mahalunge": 134, "Katraj": 132, "Ambegaon Budruk": 132, "Nigdi": 130, "Pimpri": 123, "Tingre Nagar": 122, "Pashan": 121, "Chikhali": 115, "Moshi": 111, "Tathawade": 109, "Sus": 104, "Gahunje": 98, "Dhankawadi Police Station Road": 96, "Sopan Baug": 91, "Sangamvadi": 85, "Manjari": 84, "Shivaji Nagar": 81, "New Kalyani Nagar": 81, "Erandwane": 76, "Nanded": 69, "Bhugaon": 67, "Wanwadi": 63, "Handewadi": 58, "Mohammed wadi": 58, "New Sangavi": 56, "Talegaon Dabhade": 54, "Bhosari": 54, "Manjari Budruk": 53, "Chakan": 50, "Dighi": 48, "Alandi": 47, "Magarpatta": 46, "Parvati Darshan": 45, "NIBM": 40, "NIBM Annexe": 39, "Shivane": 34, "Gultekdi": 33, "Kalewadi": 31, "Dhayari Phata": 31, "Deccan Gymkhana": 30, "Old Sangvi": 30, "Boat Club Road": 28, "Kondhwa Budruk": 27, "Bopodi": 26, "Ghorpadi": 26, "Shukrawar Peth": 25, "hingne Khurd": 25, "Narhe": 25, "Swargate": 22, "Punawale": 22, "Charholi Budruk": 22, "Gokhalenagar": 21, "Warje Malwadi": 21, "Chandan Nagar": 21, "Talwade": 21, "Loni Kalbhor": 20, "Sadashiv Peth": 20, "Somwar Peth": 19, "Anand Nagar": 18, "Agalambe": 18, "Bhegade Aali": 17, "Dattavadi": 16, "Balaji Nagar": 16, "Pune Satara Road": 16, "Bhukum": 15, "Nigdi Sector 24": 15, "Kasarwadi": 14, "Pune Station": 14, "Pradhikaran Nigdi": 14, "Mamurdi": 13, "Kothrud Depot Road": 13, "Jambhulwadi": 13, "Baner Road": 13, "Pirangut": 13, "Ambegaon Pathar": 13, "Senapati Bapat Road": 13, "Ganesh Nagar": 13, "Bopkhel": 12, "Vadgoan Sheri Rajendri Nagar": 12, "Sahakar Nagar": 12, "Ashok Nagar": 12, "Market yard": 12, "Lulla Nagar": 12, "Kasba Peth": 12, "Shaniwar Peth": 12, "Daund": 12, "Narayan Peth": 11, "Rasta Peth": 11, "Marunji": 11, "Pimpri Chinchwad": 10, "Mukund Nagar": 10, "maharshi nagar": 10}


# furnish_type
furnish_type_map = {
    "Unfurnished": 9388,
    "Semi-Furnished": 8441,
    "Furnished": 3946
}


# --- Load your trained model ---
# Option 1: Use the model from memory (retrain every time, not ideal)
# Option 2: Save your model to disk and load it here
# For now, let's assume you have trained model in 'model' variable already
# If you want to save/load:
# import joblib
# model = joblib.load("xgb_model.pkl")

# Streamlit app
st.title("Pune Rent Price Predictor")

seller_type = st.selectbox("Seller Type", list(seller_type_map.keys()))
bedroom = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=1)
layout_type = st.selectbox("Layout Type", list(layout_type_map.keys()))
property_type = st.selectbox("Property Type", list(property_type_map.keys()))
locality = st.selectbox("Locality", list(locality_map.keys()))
area = st.number_input("Area (sq.ft)", min_value=100, max_value=10000, value=650)
furnish_type = st.selectbox("Furnish Type", list(furnish_type_map.keys()))

if st.button("Predict Rent Price"):
    # Prepare input features as in your preprocessing
    seller_val = seller_type_map.get(seller_type, 0)
    layout_val = layout_type_map.get(layout_type, 0)
    property_val = property_type_map.get(property_type, 0)
    locality_val = locality_map.get(locality, 0)
    furnish_val = furnish_type_map.get(furnish_type, 0)
    
    input_features = np.array([[seller_val, bedroom, layout_val, property_val, locality_val, area, furnish_val]])
    
    # Predict
    prediction = model.predict(input_features)
    
    st.success(f"Estimated Rent Price: ₹ {prediction[0]:,.0f}")


# In[ ]:




