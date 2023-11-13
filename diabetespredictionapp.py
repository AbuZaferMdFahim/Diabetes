import numpy as np 
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler


loaded_diabetes_model = pickle.load(open('diabetes_test_model.sav','rb'))
print("Model Loaded:", loaded_diabetes_model)

def diabetes(input_data):
    print("Input Data:", input_data)
    # Convert input data to numerical types
    input_data = [float(value) for value in input_data]
    print("Numerical Input Data:", input_data)

    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    print("Numpy Array:", input_data_as_numpy_array)

    # Reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    Scaler = StandardScaler()
    Scaler.fit(input_data_reshaped)
    std_data = Scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = loaded_diabetes_model.predict(std_data)

    if prediction[0] == 0:
        return 'The Person is Not Diabetic'
    else:
        return 'The Person is Diabetic'

    
def main():
   
    # giving title
    st.title('Diabetes Prediction Web App')

    # getting the input data from the user
   

    Pregnancies = st.text_input('Number of Pregnancy')
    Glucose = st.text_input('Glucos Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function  Value')
    Age = st.text_input('Enter Your Age')

    # code for Prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__ =='__main__':
    main()
