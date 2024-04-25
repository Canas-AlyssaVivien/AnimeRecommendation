import streamlit as st
import pandas as pd
import pickle

data = pd.read_csv('processed_dataset.csv')

# Load model and preprocessing pipeline using pickle
#with open('anime_recommendation_model.pkl', 'rb') as f:
#    model = pickle.load(f)

from keras.models import load_model

model = load_model('anime_recommendation_model.h5')

with open('preprocessing_pipeline.joblib', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

def preprocess_data(input_data):
    return preprocessing_pipeline.transform(input_data)

from preprocessor import Preprocessor
preprocessor = Preprocessor()

st.title('Anime Recommendation')

anime_name = st.text_input('Enter the name of the anime')

genres = st.text_input('Enter the genres (comma-separated)')

if st.button('Predict'):
    st.write('Anime Name:', anime_name)
    st.write('Genres:', genres)

    input_data = pd.DataFrame({'genre': [genres]})
    preprocessed_data = preprocessor.transform(input_data)
    preprocessed_data = preprocessed_data.drop(columns=['genre']).astype('float32')

    st.write('PreProcessed: ', preprocessed_data)

    #preprocessed_data = preprocessor.fit(input_data)
    #preprocessed_data = preprocessor.transform(input_data)
    #st.write('2: ', preprocessed_data)
    #preprocessed_data = preprocessed_data.drop(columns=['genre']).astype('float32')

    prediction = model.predict(preprocessed_data)

    max_genre_index = prediction.argmax()
    max_genre_name = preprocessed_data.columns[max_genre_index]

    movies_with_max_genre = data[data[max_genre_name] == 1]['name']

    st.header("Movies with the Predicted Genre:")
    st.write(movies_with_max_genre)

#if st.button('Predict'):
#    st.write('Anime Name:', anime_name)
#    st.write('Genres:', genres)

#    input_dtype = model.layers[0].dtype

#    input_data = preprocess_data(pd.DataFrame({'genre': [genres]}))
#    preprocessed_data = input_data.drop(columns=['genre'])
#    preprocessed_data = preprocessed_data.astype('float32')

#    prediction = model.predict(preprocessed_data)

#    max_genre_index = prediction.argmax()
#    max_genre_name = preprocessed_data.columns[max_genre_index]

#    movies_with_max_genre = data[data[max_genre_name] == 1]['name']
#    st.header("Movies with the Predicted Genre:")
#    st.write(movies_with_max_genre)