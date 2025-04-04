import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load model and encoders
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

def safe_encode(label_encoder, value):
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        raise ValueError(f"Value '{value}' not found in encoder classes: {label_encoder.classes_}")

model, encoders = load_model()

st.title("Titanic Survival Prediction")

# User Inputs
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode inputs
try:
    sex_enc = safe_encode(encoders['Sex'], sex)
    embarked_enc = safe_encode(encoders['Embarked'], embarked)
except ValueError as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Prediction
features = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
prediction = model.predict(features)[0]
prob = model.predict_proba(features)[0][1]

# Output
if prediction == 1:
    st.success(f"Prediction: Survived (Probability: {prob:.2f})")
else:
    st.error(f"Prediction: Did not survive (Probability: {prob:.2f})")
