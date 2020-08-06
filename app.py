from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('lr_deployed_model')

cols = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    if int(data_unseen.LoanAmount) > 50000:
        return render_template('home.html',pred='Sorry , we dont offer loans more than 50000')
    else:
        prediction = predict_model(model, data=data_unseen)
        prediction = int(prediction.Label[0])
        if prediction == 1:
            message = 'Approved'
            return render_template('home.html',pred='Congratulations, your loan is  {}'.format(message))
        else:
            message = 'Rejected'
            return render_template('home.html',pred='Sorry , you loan is {}'.format(message))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
