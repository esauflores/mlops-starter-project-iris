import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def log_train_model(
    run_name: str,
    train_path: str = "data/train.csv",
    C: float = 0.01,
    solver: str = "lbfgs",
    max_iter: int = 100,
    model_path: str = "models/model.joblib",
) -> LogisticRegression:
    with mlflow.start_run(run_name=run_name, nested=True):
        # Load train set
        train_dataset = pd.read_csv(train_path)

        # Get X and Y
        y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
        X: np.ndarray = train_dataset.drop("target", axis=1).values

        # Train the model
        clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        clf.fit(X, y)

        # Log parameters
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "LogisticRegression")

        # Log metrics

        prediction = clf.predict(X)
        f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")
        accuracy: float = np.mean(prediction == y)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(clf, model_path)
        signature = mlflow.models.infer_signature(X, prediction)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="LogisticRegressionModel",
            signature=signature,
        )

    return clf


if __name__ == "__main__":
    # Load train set
    train_dataset = pd.read_csv("data/train.csv")

    # Get X and Y
    y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = train_dataset.drop("target", axis=1).values

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, "models/model.joblib")
