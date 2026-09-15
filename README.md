# House Price Prediction using Machine Learning

An end-to-end machine learning regression project that predicts house prices using property-related features. The project compares Linear Regression with Random Forest Regression and evaluates both models using MAE, RMSE, and R².

## Project Overview

The objective of this project is to build a regression model capable of predicting median house values based on housing and socioeconomic features.

The project follows a complete machine learning workflow:

1. Data loading and exploration
2. Data quality and missing-value analysis
3. Correlation analysis
4. Exploratory data visualization
5. Train-test split
6. Baseline model development
7. Advanced model development
8. Model evaluation and comparison
9. Feature importance analysis
10. Model persistence

## Dataset

The project uses the Boston Housing dataset containing:

- 506 observations
- 13 input features
- 1 target variable

### Target Variable

`medv` — Median value of owner-occupied homes.

### Features

- `crim` — Per capita crime rate
- `zn` — Proportion of residential land zoned for large lots
- `indus` — Proportion of non-retail business acres
- `chas` — Charles River dummy variable
- `nox` — Nitric oxide concentration
- `rm` — Average number of rooms
- `age` — Proportion of owner-occupied units built before 1940
- `dis` — Weighted distance to employment centers
- `rad` — Index of accessibility to radial highways
- `tax` — Property tax rate
- `ptratio` — Pupil-teacher ratio
- `b` — Proportion of Black residents
- `lstat` — Percentage of lower-status population

## Exploratory Data Analysis

Correlation analysis showed that:

- `rm` had a strong positive correlation with `medv`.
- `lstat` had the strongest negative correlation with `medv`.
- `ptratio`, `indus`, and `tax` also showed notable negative relationships with house prices.

Scatter plots were used to visualize relationships between important features and the target variable.

## Machine Learning Models

Two regression models were trained:

### 1. Linear Regression

Used as the baseline model.

### 2. Random Forest Regressor

Used as the final model because it can capture non-linear relationships and interactions between features.

## Model Evaluation

Both models were evaluated on the same unseen test set.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 3.1891 | 4.9286 | 0.6688 |
| Random Forest | **2.0413** | **2.9172** | **0.8840** |

### Final Model

The Random Forest Regressor achieved the best performance:

- **MAE:** 2.0413
- **RMSE:** 2.9172
- **R² Score:** 0.8840

The model reduced MAE and RMSE substantially compared with the Linear Regression baseline and increased the R² score from 0.6688 to 0.8840.

## Feature Importance

Random Forest feature importance was analyzed to understand which features contributed most strongly to the model's predictions.

## Example Prediction

For one test-set house:

- Actual price: **23.60**
- Predicted price: **22.82**
- Absolute error: **0.78**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Project Structure

```text
house-price-prediction-ml/
│
├── House_Price_Prediction_ML.ipynb
├── house_price_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
