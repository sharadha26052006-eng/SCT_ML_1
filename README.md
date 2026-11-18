
# House Price Prediction using Linear Regression

## SkillCraft Technology - Machine Learning Internship

### Task 01

## Project Overview
This project uses a Linear Regression machine learning algorithm to predict house prices based on the living area, number of bedrooms, and number of bathrooms.

## Objective
To develop a machine learning model that predicts house prices using important house features.

## Dataset
- Dataset: House Prices - Advanced Regression Techniques
- Source: Kaggle
- Training Records: 1460
- Total Columns: 81

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Input Features
1. GrLivArea - Above-ground living area in square feet
2. BedroomAbvGr - Number of bedrooms
3. FullBath - Number of full bathrooms

## Target Variable
SalePrice - House selling price.

## Machine Learning Algorithm
Linear Regression

## Model Evaluation

| Metric | Result |
|---|---:|
| R2 Score | 0.6341 |
| Mean Absolute Error | 35788.06 |
| Mean Squared Error | 2806426667.25 |
| Root Mean Squared Error | 52975.72 |

## Sample Prediction
For a house with:
- Living Area: 1710 square feet
- Bedrooms: 3
- Bathrooms: 2

Predicted House Price: 210209.89

## Project Files
- house_price_prediction.py - Main Python program
- train.csv - Training dataset
- test.csv - Test dataset
- prediction_results.csv - Actual and predicted prices
- prediction_graph.png - Model visualization
- README.md - Project documentation

## How to Run

1. Install Python.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Run the Python program:

```bash
python house_price_prediction.py
```

## Conclusion
The project successfully implements a Linear Regression model to predict house prices using three important features. The model is evaluated using R2 Score, MAE, MSE, and RMSE.

## Internship
SkillCraft Technology - Machine Learning Internship

Task 01: House Price Prediction
