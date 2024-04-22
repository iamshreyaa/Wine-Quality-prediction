import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier

# Function to load data
def load_data():
    return pd.read_csv("winequalityN.csv")

# Load the pre-trained Random Forest model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Main function to run the Streamlit app
def main():
    # Load data
    wine_df = load_data()

    # Load model
    classifiern = load_model()

    # Sidebar inputs
    st.sidebar.header('User Input Parameters')
    fixed_acidity = st.sidebar.slider('Fixed Acidity', float(wine_df['fixed acidity'].min()), float(wine_df['fixed acidity'].max()), float(wine_df['fixed acidity'].mean()))
    volatile_acidity = st.sidebar.slider('Volatile Acidity', float(wine_df['volatile acidity'].min()), float(wine_df['volatile acidity'].max()), float(wine_df['volatile acidity'].mean()))
    sulphates = st.sidebar.slider('Sulphates', float(wine_df['sulphates'].min()), float(wine_df['sulphates'].max()), float(wine_df['sulphates'].mean()))
    alcohol = st.sidebar.slider('Alcohol', float(wine_df['alcohol'].min()), float(wine_df['alcohol'].max()), float(wine_df['alcohol'].mean()))
    density = st.sidebar.slider('Density', float(wine_df['density'].min()), float(wine_df['density'].max()), float(wine_df['density'].mean()))

    # Adjust the column names in the input_df
    input_df = pd.DataFrame({
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'density': density
    }, index=[0])

    # Predict wine quality
    prediction_proba = classifiern.predict_proba(input_df)
    prediction = classifiern.predict(input_df)[0]

    # Determine if the wine is good or bad based on the predicted quality
    if prediction == 1:
        wine_quality = "Good"
    else:
        wine_quality = "Bad"

    # Display predictions
    st.subheader('Prediction')
    st.write('Predicted Quality:', prediction)
    st.write('Prediction Probabilities:', prediction_proba)
    st.write('Wine Quality:', wine_quality)

# Run the Streamlit app
if __name__ == '__main__':
    main()
