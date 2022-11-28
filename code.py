import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('trained_model.sav','rb'))

def prediction(input_data):
  input_data_ar = np.asarray(input_data)
  input_data_re = input_data_ar.reshape(1,-1)

  prediction = loaded_model.predict(input_data_re)

  if prediction[0] == 0:
      return 'The person\'s not gonna have a heartattack'
  else:
      return 'The person\'s gonna have a heartattack'

def main():
  st.title('Heart Attack Prediction Web App')

  age       = st.text_input('Age')
  sex       = st.text_input('Sex')
  cp        = st.text_input('Chest Pain')
  trtbps    = st.text_input('Resting Blood Pressure')
  chol      = st.text_input('Cholestoral')
  fbs       = st.text_input('Fasting Blood Sugar')
  restecg   = st.text_input('Resting Electrocardiographic results')
  thalachh  = st.text_input('Maximum Heart Rate Achieved')
  exng      = st.text_input('Exercise Induced Angina')
  oldpeak   = st.text_input('Previous Peak')
  slp       = st.text_input('Slope')
  caa       = st.text_input('Number of Major Vessels')
  thall     = st.text_input('Thal rate')

  diagnosis = ''

  if st.button('Test Result'):
    diagnosis = prediction([age, sex, cp, trtbps, chol, 
                            fbs, restecg, thalachh, exng,
                             oldpeak, slp, caa,thall])
    
  st.success(diagnosis)


if __name__ == '__main__':
  main()