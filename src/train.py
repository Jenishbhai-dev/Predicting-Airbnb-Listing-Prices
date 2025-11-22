import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from data_loader import S3DataLoader
from Air_preprocessing import AirbnbPreprocessor
from Ml_flow import MLflowModelTrainer
from Future_engineering import FeatureEngineer
load_dotenv()

# MLflow setup from .env
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "AirbnbPricePrediction")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_log_model(params):
    df = S3DataLoader()
    X, y, preprocessor = AirbnbPreprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**params)

    # Full pipeline: preprocessing + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run():
        
        # Log parameters
        mlflow.log_params(params)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        preds = pipeline.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        # Register to MLflow Model Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        mlflow.register_model(
            model_uri=model_uri,
            name="AirbnbPriceModel"
        )

        print(f"Run complete. RMSE = {rmse}")

    return rmse


if __name__ == "__main__":
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 4
    }

    train_and_log_model(params)
