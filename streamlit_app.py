import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model!')

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Dhanishthad/StreamlitML/master/dogs_cleaned.csv')
    st.write(df)

    st.write('**X**')
    X_raw = df.drop('breed', axis=1)
    st.write(X_raw)

    st.write('**y**')
    y_raw = df['breed']
    st.write(y_raw)

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='height_cm', y='body_mass_g', color='breed')

# Input features
with st.sidebar:
    st.header('Input features')
    country = st.selectbox('Country', ('UK', 'Canada', 'Germany'))
    height_cm = st.slider('Height (cm)', 30.7, 61.8, 43.9)
    weight_kg = st.slider('Weight (kg)', 11.5, 31.2, 17.2)
    tail_length_cm = st.slider('Tail length (cm)', 10.0, 38.9, 20.0)
    body_mass_g = st.slider('Body mass (g)', 32000.0, 52500.0, 42000.0)
    sex = st.selectbox('Sex', ('male', 'female'))
    
    # Create a DataFrame for the input features
    input_data = {'country': [country],
                  'height_cm': [height_cm],
                  'weight_kg': [weight_kg],
                  'tail_length_cm': [tail_length_cm],
                  'body_mass_g': [body_mass_g],
                  'sex': [sex]}
    input_df = pd.DataFrame(input_data)

with st.expander('Input features'):
    st.write('**Input data**')
    st.write(input_df)

# Data preparation
# Combine input features with sample data for demonstration
combined_df = pd.concat([input_df, X_raw], ignore_index=True)

# Encode features
encoded_df = pd.get_dummies(combined_df, columns=['country', 'sex'])
encoded_X = encoded_df.iloc[1:]  # Excluding the first row which is input data
encoded_input = encoded_df.iloc[:1]  # Only the first row for prediction

# Encode target
target_mapper = {name: idx for idx, name in enumerate(y_raw.unique())}
def target_encode(val):
    return target_mapper.get(val, -1)  # Return -1 for unknown values

encoded_y = y_raw.map(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded Input Data**')
    st.write(encoded_input)

# Model training
clf = RandomForestClassifier()
clf.fit(encoded_X, encoded_y)

# Apply model to make predictions
prediction = clf.predict(encoded_input)
prediction_proba = clf.predict_proba(encoded_input)

df_prediction_proba = pd.DataFrame(prediction_proba, columns=target_mapper.keys())

# Display predictions
st.subheader('Predicted breed probabilities')
st.dataframe(df_prediction_proba)

# Display the predicted breed
dogs_breed = list(target_mapper.keys())
st.success(f"Predicted breed: {dogs_breed[prediction[0]]}")
