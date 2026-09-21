"""
Evaluate tuned models on an untouched test set.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocess import load_data, prepare_data, split_data


RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model on the test set."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name}")
    print("-" * 50)
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")


def main():

    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)

    # Same test split used by the project
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Tuned Random Forest parameters
    random_forest = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=20,
        random_state=RANDOM_STATE
    )

    # Tuned Gradient Boosting parameters
    gradient_boosting = GradientBoostingRegressor(
        subsample=0.9,
        n_estimators=150,
        min_samples_split=10,
        min_samples_leaf=4,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )

    # Train ONLY on training data
    random_forest.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)

    print("\nTUNED MODEL TEST-SET RESULTS")
    print("=" * 60)

    evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    evaluate_model(
        gradient_boosting,
        X_test,
        y_test,
        "Tuned Gradient Boosting"
    )


if __name__ == "__main__":
    main()