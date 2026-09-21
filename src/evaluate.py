"""
Evaluate the final house price prediction model.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_data, prepare_data, split_data


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the final trained regression model.

    Args:
        model: Trained regression model.
        X_test: Test features.
        y_test: Actual target values.

    Returns:
        dict: Model evaluation metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }


def evaluate_final_model():
    """
    Load the final model and evaluate it on the test set.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    # Load dataset
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    # Recreate the same train/test split used during training
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Evaluate
    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nFinal Model Performance")
    print("=" * 60)
    print("Model: Gradient Boosting")
    print(f"MAE:      {results['MAE']:.4f}")
    print(f"RMSE:     {results['RMSE']:.4f}")
    print(f"R² Score: {results['R2 Score']:.4f}")

    return results


if __name__ == "__main__":
    evaluate_final_model()