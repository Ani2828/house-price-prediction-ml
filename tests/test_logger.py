import sqlite3

import src.logger as logger


def test_initialize_database(tmp_path, monkeypatch):
    database_path = tmp_path / "test_predictions.db"

    monkeypatch.setattr(logger, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(logger, "DATABASE_PATH", database_path)

    logger.initialize_database()

    assert database_path.exists()

    connection = sqlite3.connect(database_path)

    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='predictions'
        """
    ).fetchone()

    connection.close()

    assert tables is not None


def test_log_prediction(tmp_path, monkeypatch):
    database_path = tmp_path / "test_predictions.db"

    monkeypatch.setattr(logger, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(logger, "DATABASE_PATH", database_path)

    logger.log_prediction(
        model_name="Gradient Boosting",
        prediction=26.62
    )

    assert logger.get_prediction_count() == 1


def test_get_prediction_history(tmp_path, monkeypatch):
    database_path = tmp_path / "test_predictions.db"

    monkeypatch.setattr(logger, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(logger, "DATABASE_PATH", database_path)

    logger.log_prediction(
        model_name="Gradient Boosting",
        prediction=26.62
    )

    history = logger.get_prediction_history(limit=10)

    assert len(history) == 1
    assert history[0][1] == "Gradient Boosting"
    assert history[0][2] == 26.62