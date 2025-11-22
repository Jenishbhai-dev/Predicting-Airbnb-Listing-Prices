import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd

class MLflowModelTrainer:
    def __init__(self, experiment_name='airbnb-price-prediction'):
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def prepare_data(self, df, target_col='price', test_size=0.2, random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2_score': r2_score(y_test, predictions)
        }
        return metrics

    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        with mlflow.start_run(run_name="Linear_Regression"):
            model = LinearRegression().fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("n_features", X_train.shape[1])
            for m, v in metrics.items():
                mlflow.log_metric(m, v)
            mlflow.sklearn.log_model(model, "model")
            return model, metrics

    def train_ridge_regression(self, X_train, X_test, y_train, y_test, alpha=1.0):
        with mlflow.start_run(run_name="Ridge_Regression"):
            model = Ridge(alpha=alpha).fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("n_features", X_train.shape[1])
            for m, v in metrics.items():
                mlflow.log_metric(m, v)
            mlflow.sklearn.log_model(model, "model")
            return model, metrics

    def train_random_forest(self, X_train, X_test, y_train, y_test, n_estimators=100):
        with mlflow.start_run(run_name="Random_Forest"):
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1).fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("n_features", X_train.shape[1])
            for m, v in metrics.items():
                mlflow.log_metric(m, v)
            mlflow.sklearn.log_model(model, "model")
            return model, metrics

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1):
        with mlflow.start_run(run_name="Gradient_Boosting"):
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42).fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("n_features", X_train.shape[1])
            for m, v in metrics.items():
                mlflow.log_metric(m, v)
            mlflow.sklearn.log_model(model, "model")
            return model, metrics

    def train_xgboost(self, X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1):
        with mlflow.start_run(run_name="XGBoost"):
            model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, n_jobs=-1).fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("n_features", X_train.shape[1])
            for m, v in metrics.items():
                mlflow.log_metric(m, v)
            mlflow.xgboost.log_model(model, "model")
            return model, metrics

    def register_best_model(self, model_name="airbnb-price-predictor"):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) > 0:
            best_run = runs.loc[runs['metrics.rmse'].idxmin()]
            best_run_id = best_run['run_id']
            model_uri = f"runs:/{best_run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_name)
            return registered_model
        else:
            return None
