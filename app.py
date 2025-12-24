# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__, template_folder="template")
model = pickle.load(open("logistic.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        maxtemp = float(request.form['maxtemp'])
        temperature = float(request.form['temperature'])
        mintemp = float(request.form['mintemp'])
        dewpoint = float(request.form['dewpoint'])
        sunshine = float(request.form['sunshine'])
        windspeed = float(request.form['windspeed'])

        # Other features you want to include
        # ...

        # Create a list with the new input features
        input_lst = [maxtemp, temperature, mintemp, dewpoint, sunshine, windspeed]

        # You can include other features in input_lst

        # Make a prediction using your model
        pred = model.predict(np.array(input_lst).reshape(1, -1))

        if pred == 0:
            return render_template("sunny.html")
        else:
            return render_template("rainy.html")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
