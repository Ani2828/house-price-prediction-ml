"""
Prediction logging utilities for the House Price Prediction project.
"""

from pathlib import Path
import sqlite3
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATABASE_DIR / "predictions.db"


def initialize_database():
    """Create the prediction database and table if they do not exist."""

    DATABASE_DIR.mkdir(exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction REAL NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def log_prediction(model_name, prediction):
    """Store a prediction in the SQLite database."""

    initialize_database()

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp,
            model_name,
            prediction
        )
        VALUES (?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            model_name,
            float(prediction),
        ),
    )

    connection.commit()
    connection.close()


def get_prediction_count():
    """Return the total number of logged predictions."""

    initialize_database()

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")

    count = cursor.fetchone()[0]

    connection.close()

    return count