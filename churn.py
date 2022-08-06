import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Prediction for Churn Activity App
This app predicts the **Customer Churn**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    st = st.sidebar.slider('st', 0.0, 50.0, 20.0)
    acclen = st.sidebar.slider('acclen', 1.0, 243.0, 100.0)
    arcode = st.sidebar.slider('arcode', 408.0, 510.0, 100.0)
    intplan = st.sidebar.slider('intplan', 0.0, 1.0, 0.0)
    voice = st.sidebar.slider('voice', 0.0, 1.0, 0.0)
    nummailmes = st.sidebar.slider('intplan', 0.0, 51.0, 0.0)
    tdmin = st.sidebar.slider('tdmin', 0.0, 350.8, 100.0)
    tdcal = st.sidebar.slider('tdcal', 0.0, 165.0, 50.0)
    tdchar = st.sidebar.slider('tdchar', 0.0, 59.64, 10.0)
    temin = st.sidebar.slider('temin', 0.0, 363.7, 100.0)
    tecal = st.sidebar.slider('tecal', 0.0, 363.7, 100.0)
    tecahr = st.sidebar.slider('tecal', 0.0, 30.91, 10.0)
    tnmin = st.sidebar.slider('tnmin', 23.2, 395.0, 100.0)
    tn.cal = st.sidebar.slider('tn.cal', 33.0, 175, 100.0)
    tnchar = st.sidebar.slider('tnchar', 1.04, 17.77, 5.0)
    timin = st.sidebar.slider('timin', 0.0, 20.0, 5.0)
    tical = st.sidebar.slider('tecal', 0.0, 363.7, 100.0)
    tichar = st.sidebar.slider('tichar', 0.0, 5.4, 1.0)
    ncsc = st.sidebar.slider('ncsc', 0.0, 9.0, 1.0)
    
    data = {'st': st,
            'acclen': acclen,
            ' arcode':  arcode,
            'intplan': intplan,
             'voice' : voice,
            'nummailmes' :  nummailmes,
            'tdmin' : tdmin,
            'tdcal' : tdcal,
            'tdchar' : tdchar,
            'temin' : temin,
            'tecal' : tecal,
            'tecahr' : tecahr,
            'tnmin' : tnmin,
            'tn.cal' : tn.cal,
            'tnchar' : tnchar,
            'timin' : timin,
            'tical' : tical,
            'tichar' : tichar,
            'ncsc' : ncsc}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

churn = datasets.load_churn()
X = churn.data
Y = churn.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Churn labels and their corresponding index number')
st.write(churn.target_names)

st.subheader('Prediction')
st.write(churn.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
