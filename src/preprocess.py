"""
Data loading and preprocessing utilities for the House Price Prediction project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset URL
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"


def load_data():
    """
    Load the Boston Housing dataset.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(DATA_URL)

    return df


def prepare_data(df):
    """
    Separate features and target variable.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        tuple:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series): Target variable.
    """
    X = df.drop("medv", axis=1)
    y = df["medv"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target variable.
        test_size (float): Proportion of data used for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple:
            X_train
            X_test
            y_train
            y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test