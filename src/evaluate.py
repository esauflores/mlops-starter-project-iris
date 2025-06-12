import json
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load unique classes from the original features file
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()

    # Load test dataset
    test_dataset = pd.read_csv("data/test.csv")
    y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = test_dataset.drop("target", axis=1).values

    # Load trained model
    clf = joblib.load("models/model.joblib")

    # Make predictions
    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    cm: np.ndarray = confusion_matrix(y, prediction)
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")

    return {
        "f1_score": f1,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


def log_evaluation_metrics(
    run_name: str,
    test_path: str = "data/test.csv",
    model_path: str = "models/model.joblib",
    metrics_path: str = "data/eval.json",
    model_type: str = "LogisticRegressionModel",
) -> dict[str, Any]:
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()

    with mlflow.start_run(run_name=run_name, nested=True):
        # Load test dataset
        test_dataset = pd.read_csv(test_path)
        y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
        X: np.ndarray = test_dataset.drop("target", axis=1).values

        # Load the model
        clf = joblib.load(model_path)

        # Make predictions
        prediction: np.ndarray = clf.predict(X)

        # Calculate metrics
        cm: np.ndarray = confusion_matrix(y, prediction)
        f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")
        accuracy: float = np.mean(prediction == y)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_dict(
            {"classes": classes, "matrix": cm.tolist()}, "confusion_matrix.json"
        )

        signature = mlflow.models.infer_signature(X, prediction)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name=model_type,
            signature=signature,
        )

        metrics = {
            "f1_score": f1,
            "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()

    # Save metrics as JSON
    with open("data/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
