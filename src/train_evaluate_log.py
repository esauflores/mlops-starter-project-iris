import time

import mlflow

from evaluate import log_evaluation_metrics
from train import log_train_model

if __name__ == "__main__":
    # Mlflow setup
    MLFLOW_TRACKING_URI = "http://localhost:5001"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Set up MLFlow Client
    # client = MlflowClient()
    # # print(f"Client tracking uri: {client.tracking_uri}")
    # print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"Logistic Regression {current_time}"):
        # Train the model
        log_train_model(
            run_name="Training",
            train_path="data/train.csv",
            C=1.0,
            solver="saga",
            max_iter=100,
            model_path="models/model.joblib",
        )

        print("Model training completed. Starting evaluation...")

        # Evaluate the model
        log_evaluation_metrics(
            run_name="Evaluation",
            test_path="data/test.csv",
            model_path="models/model.joblib",
            metrics_path="data/eval.json",
        )

        print("Model evaluation completed. Run finished.")
