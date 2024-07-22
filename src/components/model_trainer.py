import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "xgb_pipeline.pkl")
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # Load the preprocessor from the file
        self.preprocessor = joblib.load(self.model_trainer_config.preprocessor_file_path)
        logging.info("Preprocessor loaded successfully")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            param_grid = {
                'regressor__n_estimators': [300, 400, 500],
                'regressor__learning_rate': [0.5],
                'regressor__max_depth': [10, 12],
                'regressor__subsample': [0.8, 0.9],
                'regressor__colsample_bytree': [0.8, 0.9],
                'regressor__min_child_weight': [15, 20],
                'regressor__reg_alpha': [0, 0.1, 0.5],
                'regressor__reg_lambda': [0, 0.1, 0.5],
            }

            xgb_regressor = XGBRegressor(random_state=73)

            xgb_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', xgb_regressor)
            ])

            xgb_search = RandomizedSearchCV(
                xgb_pipeline,
                param_distributions=param_grid,
                n_iter=20,
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1,
                error_score='raise'
            )

            xgb_search.fit(X_train, y_train)

            best_score = np.sqrt(-xgb_search.best_score_)
            best_params_ = xgb_search.best_params_
            print(f"XGBoost: {best_score}")
            print("Best parameters for XGBoost:", best_params_)

            best_model_xgb = xgb_search.best_estimator_
            y_pred_test = best_model_xgb.predict(X_test)

            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            print("Mean Absolute Error:", mae)
            print("Mean Squared Error:", mse)
            print("Root Mean Squared Error:", rmse)
            print("R^2 Score:", r2)

            # Feature importances
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            importances = best_model_xgb.named_steps['regressor'].feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
            print(feature_importances)

            # Plotting
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred_test, color='skyblue', alpha=0.5, s=50)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='navy', linestyle='--')
            plt.xlabel("Actual Fare")
            plt.ylabel("Predicted Fare")
            plt.title("Actual vs. Predicted Fares")
            plt.show()

            # Save the model
            joblib.dump(xgb_search, self.model_trainer_config.trained_model_file_path)
            print(f"Model and preprocessor saved to '{self.model_trainer_config.trained_model_file_path}'")

            # Load the model
            loaded_pipeline = joblib.load(self.model_trainer_config.trained_model_file_path)
            print(f"Model and preprocessor loaded from '{self.model_trainer_config.trained_model_file_path}'")

            # Use the loaded pipeline to make predictions
            predictions = loaded_pipeline.predict(X_test)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)
