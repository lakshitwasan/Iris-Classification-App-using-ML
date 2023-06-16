import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('SVM.pickle', 'rb'))


# Set the title of the app
st.title('Iris Flower Classification App')

# Add a brief description of the app
st.write('This app predicts the category of the plant species based on input features.')

# Define input widgets for Sepal Length, Sepal Width, Petal Length, Petal Width
sepal_length = st.slider('sepal_length', 4.3, 7.9, 5.0)
sepal_width = st.slider('sepal_width', 2.0, 4.4, 3.0)
petal_length = st.slider('petal_length', 1.0, 6.9, 4.0)
petal_width = st.slider('petal_width', 0.1, 2.5, 1.0)

# Add a button to submit the input features
submit = st.button('Submit')

# If the button is clicked, show the input features and the predicted result
if submit:
    # Create a numpy array from the input features
    input_features = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    # Make a prediction using the loaded model
    prediction = model.predict(input_features)

    # Show the input features and the predicted result
    st.write('Input features:')
    st.write(f'- Sepal Length: {sepal_length}')
    st.write(f'- Sepal Width: {sepal_width}')
    st.write(f'- Petal Length: {petal_length}')
    st.write(f'- Petal Width: {sepal_width}')

    st.write('Prediction:')
    st.write(f'- Species: {prediction[0]}')
