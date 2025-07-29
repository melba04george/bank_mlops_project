import streamlit as st
import pandas as pd
import pickle


# Load the trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Bank Marketing Prediction App")

# Create input fields for user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", options=['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 
                                   'unemployed', 'entrepreneur', 'housemaid', 'self-employed', 'student', 'unknown'])
marital = st.selectbox("Marital Status", options=['married', 'single', 'divorced'])
education = st.selectbox("Education", options=['secondary', 'tertiary', 'primary', 'unknown'])
default = st.selectbox("Default", options=['no', 'yes'])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Housing Loan", options=['yes', 'no'])
loan = st.selectbox("Personal Loan", options=['yes', 'no'])
contact = st.selectbox("Contact", options=['cellular', 'telephone'])
day = st.slider("Day", min_value=1, max_value=31, value=15)
month = st.selectbox("Month", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input("Call Duration", value=100)
campaign = st.number_input("Campaign Contacts", value=1)
pdays = st.number_input("Previous Days Contact", value=999)
previous = st.number_input("Previous Contacts", value=0)
poutcome = st.selectbox("Previous Outcome", options=['unknown', 'other', 'failure', 'success'])

# Convert categorical to numeric same as training
job_map = {'admin.':0,'technician':1,'services':2,'management':3,'retired':4,'blue-collar':5,
           'unemployed':6,'entrepreneur':7,'housemaid':8,'self-employed':9,'student':10,'unknown':11}
marital_map = {'married':0,'single':1,'divorced':2}
education_map = {'secondary':0,'tertiary':1,'primary':2,'unknown':3}
default_map = {'no':0,'yes':1}
housing_map = {'no':0,'yes':1}
loan_map = {'no':0,'yes':1}
contact_map = {'cellular':0,'telephone':1}
month_map = {'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,'jul':6,
             'aug':7,'sep':8,'oct':9,'nov':10,'dec':11}
poutcome_map = {'unknown':0,'other':1,'failure':2,'success':3}

# Create dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'job': [job_map[job]],
    'marital': [marital_map[marital]],
    'education': [education_map[education]],
    'default': [default_map[default]],
    'balance': [balance],
    'housing': [housing_map[housing]],
    'loan': [loan_map[loan]],
    'contact': [contact_map[contact]],
    'day': [day],
    'month': [month_map[month]],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome_map[poutcome]]
})

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        result = "Subscribed" if prediction == 1 else "Not Subscribed"
        st.success(f"Prediction: {result}")
        st.info(f"Prediction Confidence: {proba:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


