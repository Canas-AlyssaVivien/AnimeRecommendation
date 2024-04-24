import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from preprocessor import Preprocessor

data = pd.read_csv('processed_dataset.csv')

model = load('anime_recommendation_model.pkl')
model.summary()

def preprocess_data(input_data):
    return preprocessing_pipeline.transform(input_data)

st.title('Anime Recommendation')

anime_name = st.text_input('Enter the name of the anime')

genres = st.text_input('Enter the genres (comma-separated)')

model.summary()

if st.button('Predict'):
    st.write('Anime Name:', anime_name)
    st.write('Genres:', genres)

    input_dtype = model.layers[0].dtype

    input_data = preprocess_data(pd.DataFrame({'genre': [genres]}))
    preprocessed_data = input_data.drop(columns=['genre'])
    preprocessed_data = preprocessed_data.astype('float32')

    prediction = model.predict(preprocessed_data)

    max_genre_index = prediction.argmax()
    max_genre_name = preprocessed_data.columns[max_genre_index]

    movies_with_max_genre = data[data[max_genre_name] == 1]['name']
    st.header("Movies with the Predicted Genre:")
    st.write(movies_with_max_genre)