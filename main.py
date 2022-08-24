from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
cors=CORS(app)
model = pickle.load(open("RegressionModel.pkl", 'rb'))
ssc = pd.read_csv("ssc_marks.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    marks = sorted(ssc['Marks'].unique())
    caste = sorted(ssc['Caste'].unique())
    school = sorted(ssc['School'].unique())
    return render_template("index.html", marks=marks, caste=caste, school=school)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    mark = request.form.get("marks")
    caste = request.form.get("caste")
    school = request.form.get("school")

    prediction = model.predict(pd.DataFrame(columns=["Marks", "Caste", "School"], data=np.array([mark, caste, school]).reshape(1,3)))


    result=str(np.round(prediction[0], 2))

    return "<h1>Your Chances are </h1>" + result +"%"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
