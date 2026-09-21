"""
Evaluate regression models for the House Price Prediction project.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train import train_models


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained regression model.

    Args:
        model: Trained regression model.
        X_test: Test features.
        y_test: Actual target values.
        model_name (str): Name of the model.

    Returns:
        dict: Model evaluation metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    return results


def evaluate_models():
    """
    Train models and evaluate their performance.
    """

    (
        linear_model,
        random_forest_model,
        X_test,
        y_test
    ) = train_models()

    linear_results = evaluate_model(
        linear_model,
        X_test,
        y_test,
        "Linear Regression"
    )

    random_forest_results = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    results = [
        linear_results,
        random_forest_results
    ]

    print("\nModel Performance")
    print("=" * 60)

    for result in results:
        print(f"\n{result['Model']}")
        print(f"MAE:      {result['MAE']:.4f}")
        print(f"RMSE:     {result['RMSE']:.4f}")
        print(f"R² Score: {result['R2 Score']:.4f}")

    return results


if __name__ == "__main__":
    evaluate_models()