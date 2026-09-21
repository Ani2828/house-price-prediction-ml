"""
Compare multiple regression models using 5-fold cross-validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.preprocess import load_data, prepare_data


RANDOM_STATE = 42
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_models():
    """
    Create the regression models used for comparison.

    Returns:
        dict: Model name and model object.
    """

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=10000))
        ]),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    }

    return models


def compare_models():
    """
    Compare all models using 5-fold cross-validation.

    Returns:
        pandas.DataFrame: Cross-validation results.
    """

    df = load_data()
    X, y = prepare_data(df)

    models = get_models()

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for name, model in models.items():

        mae_scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_absolute_error"
        )

        mse_scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error"
        )

        r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2"
        )

        rmse_scores = np.sqrt(mse_scores)

        results.append({
            "Model": name,
            "MAE": mae_scores.mean(),
            "RMSE": rmse_scores.mean(),
            "R2 Score": r2_scores.mean()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    return results_df


if __name__ == "__main__":

    results = compare_models()

    print("\n5-Fold Cross-Validation Results")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save results
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    results_path = reports_dir / "model_comparison.csv"

    results.to_csv(
        results_path,
        index=False
    )

    print(f"\nResults saved to: {results_path}")
        # Create model comparison chart
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.bar(
        results["Model"],
        results["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model Comparison - 5-Fold Cross-Validation")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    chart_path = figures_dir / "model_comparison_rmse.png"

    plt.savefig(chart_path, dpi=300)

    plt.close()

    print(f"Chart saved to: {chart_path}")