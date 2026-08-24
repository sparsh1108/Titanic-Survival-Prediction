import streamlit as st
import requests

Pclass = st.selectbox("Passenger Class",[1,2,3])

Sex = st.selectbox("Sex",["Male","Female"])

Age = st.number_input("Age",0,100)

Fare = st.number_input("Fare")

Cabin = st.text_input("Cabin","Unknown")

Embarked = st.selectbox("Embarked",["S","C","Q"])

with_family = st.number_input("with_family")



if st.button("Predict"):
    data = {
        "Pclass":Pclass,
        "Sex":Sex,
        "Age":Age,
        "Fare":Fare,
        "Cabin":Cabin,
        "Embarked":Embarked,
        "with_family":with_family
    }

    response = requests.post("https://titanic-survival-prediction-2-4moc.onrender.com/predict",json=data)


    result = response.json()


    st.success(result["prediction"])
    
