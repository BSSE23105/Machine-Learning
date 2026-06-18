import os
import joblib
import pandas as pd
from flask import Flask, request, render_template

app =Flask(__name__)

# load model
if os.path.exists("model.joblib"):
    mdl= joblib.load("model.joblib")
    print("model loaded ok")
else:
    mdl =None
    print("no model found")

cols =["age", "income", "hour", "leak_feature"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if mdl is None:
        return render_template("index.html", error="model not loaded")

    try:
        vals =[
            float(request.form["age"]),
            float(request.form["income"]),
            float(request.form["hour"]),
            float(request.form["leak_feature"]),
        ]

        df = pd.DataFrame([vals], columns= cols)
        result =mdl.predict(df)
        out= int(result[0])
        return render_template("index.html", prediction =out)

    except Exception as e:
        return render_template("index.html", error= str(e))

if __name__ =="__main__":
    app.run(host= "0.0.0.0", port=5000, debug=True)
