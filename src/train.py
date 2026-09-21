"""
Train the final house price prediction model.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from src.preprocess import load_data, prepare_data, split_data


# Reproducibility
RANDOM_STATE = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_final_model():
    """
    Train the tuned Gradient Boosting model and save it.
    """

    # Load dataset
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Tuned Gradient Boosting configuration
    model = GradientBoostingRegressor(
        subsample=0.9,
        n_estimators=150,
        min_samples_split=10,
        min_samples_leaf=4,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )

    # Train
    model.fit(X_train, y_train)

    # Save final model
    model_path = MODEL_DIR / "house_price_model.pkl"

    joblib.dump(model, model_path)

    print("Final model trained successfully.")
    print("Model: Gradient Boosting")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_final_model()