import numpy as np
import pickle
import streamlit as st

loading_model = pickle.load(open("trained_model.pkl", 'rb'))


def diabetes_pred(input_data):
    #to will giv e reshape
    x_arra = np.asarray(input_data)
    x_reshape = x_arra.reshape(1, -1)
    predt_y = loading_model.predict(x_reshape)
    print(predt_y)
    if predt_y[0] == 0:
        return "The person is not diabetes"
    else:
        return "The person is diabetes"


#using and creating stramlit

def main():

    #giving title
    st.title("DIABETES PREDICTION")

    #giting input values 
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Number of Glucose")
    BloodPressure = st.text_input("BloodPressure Level")
    SkinThickness = st.text_input("SkinThickness values ")
    Insulin = st.text_input("Insulin Level")
    BMI_i = st.text_input("BMI(body mass index)")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
    age = st.text_input("Age of you")

    # code of prediction 
    diagnosis = ""

    #creating button
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI_i, DiabetesPedigreeFunction, age])
    
    st.success(diagnosis)



if __name__=='__main__':
    main()















