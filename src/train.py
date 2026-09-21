"""
Train regression models for the House Price Prediction project.
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import load_data, prepare_data, split_data


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_models():
    """
    Load data, split it, and train Linear Regression
    and Random Forest models.

    Returns:
        tuple:
            linear_model
            random_forest_model
            X_test
            y_test
    """

    # Load dataset
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Random Forest
    random_forest_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    random_forest_model.fit(X_train, y_train)

    # Save Random Forest model
    model_path = MODEL_DIR / "house_price_model.pkl"

    joblib.dump(random_forest_model, model_path)

    print("Models trained successfully.")
    print(f"Random Forest model saved to: {model_path}")

    return (
        linear_model,
        random_forest_model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    train_models()