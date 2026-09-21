from src.predict import load_model, predict_house_price


def test_model_loads():
    model = load_model()

    assert model is not None


def test_prediction_returns_number():
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

    prediction = predict_house_price(sample_house)

    assert isinstance(prediction, float)


def test_prediction_is_positive():
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

    prediction = predict_house_price(sample_house)

    assert prediction > 0