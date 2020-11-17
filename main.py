import numpy as np
import pickle
from flask import Flask, request, render_template
import joblib
# Load ML model
model = pickle.load(open('randomforest_classifier_model.pkl', 'rb'))
scaler =joblib.load("scaler.save")
# Create application
app = Flask(__name__)

import pandas as pd

# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')


# Bind predict function to URL
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Put all form entries values in a list
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    cp = float(request.form['cp'])
    fbs = float(request.form['fbs'])
    oldpeak=float(request.form['oldpeak'])
    slope=float(request.form['slope'])
    ca=float(request.form['ca'])
    thal=float(request.form['thal'])
    
    scaled_data=scaler.transform([[trestbps,chol,thalach,age,oldpeak]])
    trestbps1=scaled_data[0][0]
    chol1 = scaled_data[0][1]
    thalach1= scaled_data[0][2]
    age1 = scaled_data[0][3]
    oldpeak1 = scaled_data[0][4]

    array_features = np.array([age1, sex, cp, trestbps1, chol1, fbs, restecg,
                               thalach1, exang, oldpeak1, slope, ca, thal]).reshape(1, -1)

    # Convert features to array
    # Predict features
    prediction = model.predict(array_features)
    output = prediction
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('result.html',
                               result='The patient is not likely to have heart disease!',output=1)
    else:
        return render_template('result.html',
                               result='The patient is likely to have heart disease!',output=0)


if __name__ == '__main__':
    # Run the application
    app.run(debug=True)

