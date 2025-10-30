import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('regression.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


st.title("Customer Salary Prediction ")

geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 1, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Create input data DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Concatenate with geography encoding
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the features
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

# Display results with proper formatting
st.write(f'Predicted Estimated Salary: {prediction_salary:.2f}')
