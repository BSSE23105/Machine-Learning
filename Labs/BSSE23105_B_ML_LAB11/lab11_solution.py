# ============================================================
# Imports
# ============================================================

# Standard data / ML libraries
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import uvicorn

# FastAPI framework — HTTPException is available for use in the API section
from fastapi import FastAPI, HTTPException


# ============================================================
# ML Pipeline
# ============================================================

# Accuracy threshold used by tests to validate pipeline quality
expected_accuracy = 0.8


def load_data():
    """
    Load the dataset from the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe containing the data.
    """
    df= pd.read_csv("lab10_data.csv")
    return df


def preprocess_data(df):
    """
    Preprocess the input dataframe by cleaning and preparing features and target.

    Args:
        df (pd.DataFrame): The raw dataframe loaded from the CSV.

    Returns:
        tuple: A tuple containing (X, y) where X is the feature matrix and y is the target vector.
    """
    # drop city column since its categorical
    df = df.drop("city", axis=1)
    X= df.drop("target", axis =1)
    y = df["target"]
    return (X, y)


def train_model(X, y):
    """
    Train a machine learning model using the provided features and target.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix.
        y (pd.Series or np.ndarray): The target vector.

    Returns:
        object: The trained model object.
    """
    clf= DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf


def evaluate_model(model, X, y):
    """
    Evaluate the trained model on the given data and return the accuracy.

    Args:
        model (object): The trained model.
        X (pd.DataFrame or np.ndarray): The feature matrix for evaluation.
        y (pd.Series or np.ndarray): The true target values.

    Returns:
        float: The accuracy score of the model.
    """
    preds= model.predict(X)
    acc = accuracy_score(y, preds)
    return acc


def run_pipeline():
    """
    Run the complete ML pipeline: load data, preprocess, train, evaluate, and save the model.

    Returns:
        float: The accuracy of the trained model on the training data.
    """
    df = load_data()
    X, y= preprocess_data(df)

    mlflow.set_experiment("lab11_pipeline")
    with mlflow.start_run():
        mdl= train_model(X, y)
        acc = evaluate_model(mdl, X, y)
        # log params
        mlflow.log_param("max_depth", mdl.get_params()["max_depth"])
        mlflow.log_param("criterion", mdl.get_params()["criterion"])
        # log metric
        mlflow.log_metric("accuracy", acc)
        # log the model artifact
        mlflow.sklearn.log_model(mdl, "model")

    joblib.dump(mdl, "model.joblib")
    print(f"model saved. accuracy: {acc}")
    return acc


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI()

# load saved model if available
if os.path.exists("model.joblib"):
    model =joblib.load("model.joblib")
else:
    model = None


@app.get("/")
def home():
    # Returns a simple status message to confirm the API is reachable.
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    Run inference for a single sample.

    Expected JSON body:
        { "features": [age, income, hour, leak_feature] }

    Returns:
        dict: { "prediction": 0 or 1 }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if "features" not in data:
        raise HTTPException(status_code=422, detail="Missing 'features' key in request body")
    cols = ["age", "income", "hour", "leak_feature"]
    inp= pd.DataFrame([data["features"]], columns=cols)
    result = model.predict(inp)
    return {"prediction": int(result[0])}


# main block to run and test everything
if __name__== "__main__":

    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    # --- task 2 output ---
    print("task 2 - modularization\n")

    df = load_data()
    print("loaded the dataset")
    print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
    print(f"column names: {list(df.columns)}")

    X, y= preprocess_data(df)
    print(f"\npreprocessed the data")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"features used: {list(X.columns)}")

    clf= train_model(X, y)
    print(f"\ntrained the decision tree")
    print(f"max_depth = {clf.get_params()['max_depth']}")
    print(f"criterion = {clf.get_params()['criterion']}")

    acc =evaluate_model(clf, X, y)
    print(f"\nmodel accuracy: {acc}")
    if acc>= expected_accuracy:
        print(f"this is above the threshold of {expected_accuracy}, so we are good")
    else:
        print(f"accuracy is below {expected_accuracy}, need to improve")


    # --- task 3 output ---
    print("\n\ntask 3 - mlflow logging\n")

    print("running the full pipeline now (with mlflow)...")
    final_acc= run_pipeline()
    print(f"pipeline done, final accuracy = {final_acc}")


    # --- task 4 output ---
    print("\n\ntask 4 - serialization\n")

    print("testing if saved model works...")
    loaded_mdl =joblib.load("model.joblib")
    test_row = pd.DataFrame([X.iloc[0].values], columns=X.columns)
    pred = loaded_mdl.predict(test_row)
    print(f"sample features: {[float(round(v,4)) for v in X.iloc[0].values]}")
    print(f"prediction: {pred[0]}")
    print(f"actual label: {y.iloc[0]}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

