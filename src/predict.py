"""
Make house price predictions using the trained model.
"""

from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"


# Features expected by the model
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
    Load the trained model from disk.

    Returns:
        Trained machine learning model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )

    return joblib.load(MODEL_PATH)


def validate_features(features):
    """
    Validate prediction input.

    Args:
        features (dict): Input feature dictionary.

    Raises:
        ValueError: If features are missing, unexpected,
                    or contain invalid values.
    """

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    unexpected_features = [
        feature
        for feature in features
        if feature not in FEATURES
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {unexpected_features}"
        )

    for feature in FEATURES:
        value = features[feature]

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

        if pd.isna(value):
            raise ValueError(
                f"Feature '{feature}' cannot be empty."
            )


def predict_house_price(features):
    """
    Predict house price from input features.

    Args:
        features (dict): House feature values.

    Returns:
        float: Predicted house price.
    """

    validate_features(features)

    model = load_model()

    input_data = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return float(prediction)


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

    print(
        f"Predicted house price: {predicted_price:.2f}"
    )