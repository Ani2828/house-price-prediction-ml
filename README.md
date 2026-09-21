# 🏠 House Price Prediction using Machine Learning

An end-to-end machine learning regression project that predicts median house values from housing and socioeconomic features.

The project evolved from a baseline Linear Regression/Random Forest experiment into a reproducible ML pipeline with:

* Multiple model comparison
* 5-fold cross-validation
* Hyperparameter tuning
* Test-set evaluation
* Automated testing
* Model persistence
* Streamlit web application
* Public cloud deployment

## 🚀 Live Demo

**Try the deployed application:**

[House Price Prediction — Live Demo](https://house-price-prediction-ml-1234.streamlit.app?utm_source=chatgpt.com)

The application allows users to enter housing characteristics and receive a model-generated price estimate.

---

## 📌 Project Overview

The objective is to build a regression system capable of estimating median house values from 13 input features.

The project follows an end-to-end machine learning workflow:

```text
Data
  ↓
Data Exploration & Analysis
  ↓
Preprocessing
  ↓
Train/Test Split
  ↓
Multiple Model Comparison
  ↓
5-Fold Cross-Validation
  ↓
Hyperparameter Tuning
  ↓
Final Test Evaluation
  ↓
Model Persistence
  ↓
Automated Testing
  ↓
Streamlit Application
  ↓
Cloud Deployment
```

---

## 📊 Dataset

This project uses the **Boston Housing dataset** containing:

* **506 observations**
* **13 input features**
* **1 target variable**

### Target Variable

`medv` — Median value of owner-occupied homes, expressed in thousands of US dollars in the original dataset.

### Features

| Feature   | Description                                              |
| --------- | -------------------------------------------------------- |
| `crim`    | Per-capita crime rate                                    |
| `zn`      | Proportion of residential land zoned for large lots      |
| `indus`   | Proportion of non-retail business acres                  |
| `chas`    | Charles River proximity indicator                        |
| `nox`     | Nitric oxide concentration                               |
| `rm`      | Average number of rooms per dwelling                     |
| `age`     | Proportion of owner-occupied units built before 1940     |
| `dis`     | Weighted distance to employment centers                  |
| `rad`     | Accessibility to radial highways                         |
| `tax`     | Property tax rate                                        |
| `ptratio` | Pupil-teacher ratio                                      |
| `b`       | Historical dataset feature derived from the `B` variable |
| `lstat`   | Percentage of lower-status population                    |

---

## 🔍 Exploratory Data Analysis

The original exploratory analysis examined:

* Dataset structure
* Missing values
* Feature distributions
* Correlations
* Relationships between important features and the target
* Feature importance from tree-based models

Notable relationships observed during EDA included:

* `rm` showed a strong positive relationship with `medv`.
* `lstat` showed a strong negative relationship with `medv`.
* `ptratio`, `indus`, and `tax` also showed notable relationships with the target.

Scatter plots and correlation analysis were used to investigate these relationships.

---

# 🤖 Machine Learning Models

The project compares six regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting

This provides a broader comparison than evaluating only one baseline and one tree-based model.

---

## 🔄 5-Fold Cross-Validation

The models were evaluated using **5-fold cross-validation** with shuffled folds and a fixed random state for reproducibility.

Initial cross-validation results:

| Model             |    MAE |   RMSE |     R² |
| ----------------- | -----: | -----: | -----: |
| Gradient Boosting | 2.2152 | 3.0658 | 0.8854 |
| Random Forest     | 2.1966 | 3.2494 | 0.8710 |
| Ridge Regression  | 3.3832 | 4.8426 | 0.7152 |
| Linear Regression | 3.3885 | 4.8428 | 0.7152 |
| Lasso Regression  | 3.3673 | 4.8940 | 0.7090 |
| Decision Tree     | 3.1897 | 4.9302 | 0.6997 |

The experiment results are stored in:

```text
reports/model_comparison.csv
```

A visualization of RMSE comparison is available at:

```text
reports/figures/model_comparison_rmse.png
```

---

# ⚙️ Hyperparameter Tuning

Randomized hyperparameter search with 5-fold cross-validation was performed for:

* Random Forest
* Gradient Boosting

### Tuned Random Forest

Best configuration found:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
```

Cross-validation RMSE:

```text
3.2682
```

### Tuned Gradient Boosting

Best configuration found:

```text
n_estimators = 150
learning_rate = 0.1
max_depth = 3
min_samples_split = 10
min_samples_leaf = 4
subsample = 0.9
```

Cross-validation RMSE:

```text
2.9691
```

---

# 🏆 Final Model

The final model used by the application is a tuned:

```text
GradientBoostingRegressor
```

The final configuration was evaluated on an **untouched test set** after hyperparameter selection.

### Final Test-Set Performance

| Metric |      Score |
| ------ | ---------: |
| MAE    | **1.9301** |
| RMSE   | **2.6550** |
| R²     | **0.9039** |

These metrics represent the performance of the final tuned model on the project's held-out test set.

The trained model is saved as:

```text
models/house_price_model.pkl
```

---

# 🧪 Automated Testing

The project includes automated tests using **pytest**.

Current test suite:

```text
6 tests
```

Latest test result:

```text
6 passed
```

Run the tests with:

```bash
python -m pytest -q
```

---

# 🌐 Streamlit Application

A Streamlit interface was developed on top of the trained model.

The application provides:

* Property feature inputs
* Input validation
* House price prediction
* Model performance information
* Interactive user interface
* Error handling

Application architecture:

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
src/predict.py
 │
 ▼
Input Validation
 │
 ▼
house_price_model.pkl
 │
 ▼
Gradient Boosting Model
 │
 ▼
Predicted House Value
```

---

# ☁️ Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

Live application:

[Open the deployed House Price Predictor](https://house-price-prediction-ml-1234.streamlit.app?utm_source=chatgpt.com)

The deployment is connected to the GitHub repository so application updates can be deployed from the repository.

---

# 📁 Project Structure

```text
house-price-prediction-ml/
│
├── app/
│   └── app.py
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── House_Price_Prediction_ML.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│       └── model_comparison_rmse.png
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model_comparison.py
│   ├── tune_models.py
│   └── tuned_evaluate.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_prediction.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Testing

* Pytest

### Application

* Streamlit

### Development

* Visual Studio Code
* Git
* GitHub

### Deployment

* Streamlit Community Cloud

---

# ▶️ Run the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/Ani2828/house-price-prediction-ml.git
```

```bash
cd house-price-prediction-ml
```

## 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Train the final model

```bash
python -m src.train
```

## 4. Generate a prediction

```bash
python -m src.predict
```

## 5. Compare machine learning models

```bash
python -m src.model_comparison
```

## 6. Run hyperparameter tuning

```bash
python -m src.tune_models
```

## 7. Evaluate the tuned models

```bash
python -m src.tuned_evaluate
```

## 8. Run automated tests

```bash
python -m pytest -q
```

## 9. Launch the Streamlit application

```bash
python -m streamlit run app/app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

# 🔬 Reproducibility

The project uses fixed random states where appropriate:

```python
random_state = 42
```

This helps make model training, cross-validation, and evaluation reproducible.

The ML workflow is separated into reusable Python modules rather than relying exclusively on a Jupyter notebook.

---

# ⚠️ Dataset Limitations

The Boston Housing dataset is a **historical dataset** and has important limitations.

In particular:

* It is relatively small by modern machine-learning standards.
* Some features reflect historical socioeconomic and demographic conditions.
* The dataset should not be interpreted as a modern real-estate valuation dataset.
* The model is intended as a machine-learning demonstration and educational project, not as a real-world property valuation system.

The reported metrics therefore describe performance on this specific dataset and evaluation setup; they should not be interpreted as evidence of real-world housing-price accuracy.

---

# 📈 Future Improvements

Potential extensions include:

* Feature engineering
* More extensive hyperparameter optimization
* Residual and error analysis
* Prediction confidence/uncertainty estimation
* Model explainability using SHAP
* FastAPI model-serving endpoint
* Docker containerization
* CI/CD with GitHub Actions
* Experiment tracking
* Modern real-estate datasets
* Monitoring for model drift

---

# 👨‍💻 Author

**Ani2828**

Built as an end-to-end machine learning engineering project covering model development, evaluation, testing, application development, and deployment.
