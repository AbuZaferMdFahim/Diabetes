import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

# Loading a saved model
loaded_diabetes_model = pickle.load(open('C:/Users/one/Desktop/Diabetes-main/diabetes_test_model.sav', 'rb'))
print("Model Loaded:", loaded_diabetes_model)

# Input data
input_data = (10, 115, 0, 0, 0, 35.3, 0.134, 29)

# Convert input data to float
input_data = [float(value) for value in input_data]
print("Numerical Input Data:", input_data)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)
print("Numpy Array:", input_data_as_numpy_array)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
print("Reshaped Numpy Array:", input_data_reshaped)

Scaler = StandardScaler()
Scaler.fit(input_data_reshaped)
std_data = Scaler.transform(input_data_reshaped)

# Make prediction
prediction = loaded_diabetes_model.predict(std_data)
print("Prediction:", prediction)

if prediction[0] == 0:
    print('The Person is Not Diabetic')
else:
    print('The Person is Diabetic')
