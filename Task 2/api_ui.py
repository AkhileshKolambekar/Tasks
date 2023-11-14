import streamlit as st
import requests

st.title("Task 2")
co = st.number_input('Enter CO:')
s1 = st.number_input('Enter S1:')
nmhc = st.number_input('Enter NMHC:')
c6h6 = st.number_input('Enter C6H6:')
nox = st.number_input('Enter NOx:')
s3 = st.number_input('Enter S3:')
no2 = st.number_input('Enter NO2:')
s4 = st.number_input('Enter S4:')
s5 = st.number_input('Enter S5:')
t = st.number_input('Enter T:')
rh = st.number_input('Enter RH:')
ah = st.number_input('Enter AH:')

if st.button("Predict"):
    url = 'http://localhost:5000/predict'
    data = {'CO':co,'S1':s1,'NMHC':nmhc,'C6H6':c6h6,'NOx':nox,'S3':s3,'NO2':no2,'S4':s4,'S5':s5,'T':t,'RH':rh,'AH':ah}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        prediction = response.json()['Prediction']
        st.success(f'Predicted value is : {prediction}')

    else:
        st.error('Error in making prediction. Please check your inputs.')