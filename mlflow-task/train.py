import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_experiments():
    # 1. Load Dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target

    # 2. Split Dataset (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define hyperparameter configurations for the 3 runs
    experiments = [
        {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
        {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
        {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
    ]

    # Set MLflow experiment name
    mlflow.set_experiment("House_Price_Prediction_MLflow")

    for exp in experiments:
        with mlflow.start_run(run_name=exp["run_name"]):
            # Train model
            model = GradientBoostingRegressor(
                max_depth=exp["max_depth"],
                learning_rate=exp["learning_rate"],
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Evaluate predictions
            predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)

            # Log Hyperparameters
            mlflow.log_param("max_depth", exp["max_depth"])
            mlflow.log_param("learning_rate", exp["learning_rate"])

            # Log Metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Log Model Artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(
                f"{exp['run_name']} Finished -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
            )


if __name__ == "__main__":
    run_experiments()