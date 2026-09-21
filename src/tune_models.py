"""
Hyperparameter tuning for regression models.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold
)

from src.preprocess import load_data, prepare_data


RANDOM_STATE = 42
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def tune_random_forest(X, y, cv):
    """
    Tune Random Forest hyperparameters.
    """

    model = RandomForestRegressor(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X, y)

    return search


def tune_gradient_boosting(X, y, cv):
    """
    Tune Gradient Boosting hyperparameters.
    """

    model = GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X, y)

    return search


def main():

    # Load data
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    # Cross-validation strategy
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("Tuning Random Forest...")
    rf_search = tune_random_forest(X, y, cv)

    print("\nTuning Gradient Boosting...")
    gb_search = tune_gradient_boosting(X, y, cv)

    # Display results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)

    print("\nRandom Forest")
    print("-" * 70)
    print("Best RMSE:",
          f"{-rf_search.best_score_:.4f}")
    print("Best parameters:")
    print(rf_search.best_params_)

    print("\nGradient Boosting")
    print("-" * 70)
    print("Best RMSE:",
          f"{-gb_search.best_score_:.4f}")
    print("Best parameters:")
    print(gb_search.best_params_)

    # Save tuned models
    rf_path = MODEL_DIR / "random_forest_tuned.pkl"
    gb_path = MODEL_DIR / "gradient_boosting_tuned.pkl"

    joblib.dump(
        rf_search.best_estimator_,
        rf_path
    )

    joblib.dump(
        gb_search.best_estimator_,
        gb_path
    )

    print("\nModels saved:")
    print(rf_path)
    print(gb_path)


if __name__ == "__main__":
    main()