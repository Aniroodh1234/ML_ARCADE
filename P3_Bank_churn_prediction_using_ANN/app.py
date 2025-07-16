import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import pandas as pd
import streamlit as st
import numpy as np


with open('label encoder_gender.pkl','rb') as file:
    label_gender = pickle.load(file)
with open('onehot encoder_geo.pkl','rb') as file:
    onehot_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model('model.h5')

## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_gender.transform([gender])[0]],  ## performing label encoding on gender
    'Geography':[geography],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


## converting dictionary to dataframes
input_df = pd.DataFrame([input_data])


## performing onehot encoding on Geography

onehot_array_geo = onehot_geo.transform([['Geography']]).toarray()
onehot_df_geo = pd.DataFrame(onehot_array_geo,columns=onehot_geo.get_feature_names_out[['Geography']])

pd.concat([input_df.drop('Geography',axis=1),onehot_df_geo],axis=1 )

## scaling data
scaled_data = scaler.transform(input_df)

## predict the churn
predictions = model.predict(scaled_data)
predictions_proba = predictions[0][0]

st.write(f'Churn probability: {predictions_proba:.2f}')


if predictions_proba>0.5:
    st.write('The custome is likely to churn')
else:
    st.write('The customer is unlikely to churn')