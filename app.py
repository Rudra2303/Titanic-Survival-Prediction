import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("Titanic Survival Prediction")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert categorical variables
sex = 1 if sex == "Male" else 0

if embarked == "C":
    embarked = 0
elif embarked == "Q":
    embarked = 1
else:
    embarked = 2

# Additional features
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_per_person = fare / family_size
age_group = age // 10

# Prediction
if st.button("Predict"):
    features = np.array(
        [
            [
                pclass,
                sex,
                age,
                age_group,
                sibsp,
                parch,
                family_size,
                is_alone,
                fare,
                fare_per_person,
                embarked,
            ]
        ]
    )

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
