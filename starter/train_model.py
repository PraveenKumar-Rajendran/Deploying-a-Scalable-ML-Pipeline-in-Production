# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pickle
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
import numpy as np
import pandas as pd
from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")

def main():
    # Add code to load in the data.
    data = pd.read_csv("data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )


    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save the model and label binarizer to disk.
    model_filename = "model/model.pkl"
    lb_filename = "model/label_binarizer.pkl"

    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(lb_filename, "wb") as lb_file:
        pickle.dump(lb, lb_file)

    # Validate the model.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Make predictions on the test data.
    preds = inference(model, X_test)

    # Compute the model metrics.
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # pretty print the metrics in multiple colors
    cprint(f"Precision: {precision:.2f}", "green")
    cprint(f"Recall: {recall:.2f}", "light_blue")
    cprint(f"F1: {fbeta:.2f}", "magenta")

if __name__ == "__main__":
    main()