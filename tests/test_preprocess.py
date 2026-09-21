from src.preprocess import load_data, prepare_data, split_data


def test_load_data():
    df = load_data()

    assert df.shape == (506, 14)
    assert "medv" in df.columns


def test_prepare_data():
    df = load_data()

    X, y = prepare_data(df)

    assert X.shape == (506, 13)
    assert y.shape == (506,)
    assert "medv" not in X.columns


def test_split_data():
    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert X_train.shape == (404, 13)
    assert X_test.shape == (102, 13)

    assert y_train.shape == (404,)
    assert y_test.shape == (102,)