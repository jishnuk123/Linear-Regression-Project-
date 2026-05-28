# Linear Regression with PyTorch

A linear regression model built from scratch using PyTorch, trained on the 
California Housing dataset to predict house prices.

**[Live App](https://california-house-predictor-jk.streamlit.app)**

## Project Overview

This project demonstrates a full machine learning workflow:
- Data loading and preprocessing
- Building a custom PyTorch model
- Training with gradient descent
- Evaluating with RMSE and R²
- Deploying as an interactive web app

## Results

| Metric | Value |
|---|---|
| Test R² | 0.60 |
| Test RMSE | $72,671 |

R² of 0.60 on real housing data is expected for linear regression — 
the dataset has non-linear relationships that a more complex model 
would capture better. This model serves as a strong baseline.

## Files

| File | Description |
|---|---|
| `LinearRegressionModel.ipynb` | Synthetic data — learning the fundamentals |
| `california_housing_linearregression.ipynb` | Real California Housing dataset |
| `app.py` | Streamlit web app |
| `results.png` | Training visualizations |

## Tech Stack

- PyTorch
- Scikit-learn
- Streamlit
- Matplotlib
- NumPy

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Concepts Demonstrated

- Custom `nn.Module` model architecture
- Manual training loop with backpropagation
- Feature standardisation without data leakage
- Train/val/test split evaluation
- Model deployment with Streamlit