"""
Make house price predictions using the trained Random Forest model.
"""

from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"


# Feature order used during training
FEATURES = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat"
]


def load_model():
    """
    Load the trained Random Forest model.

    Returns:
        Trained model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run the training pipeline first."
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_house_price(features):
    """
    Predict the house price for a single house.

    Args:
        features (dict): House feature values.

    Returns:
        float: Predicted house price.
    """

    model = load_model()

    input_data = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return prediction


if __name__ == "__main__":

    sample_house = {
        "crim": 0.00632,
        "zn": 18.0,
        "indus": 2.31,
        "chas": 0,
        "nox": 0.538,
        "rm": 6.575,
        "age": 65.2,
        "dis": 4.0900,
        "rad": 1,
        "tax": 296,
        "ptratio": 15.3,
        "b": 396.90,
        "lstat": 4.98
    }

    predicted_price = predict_house_price(sample_house)

    print(f"Predicted house price: {predicted_price:.2f}")