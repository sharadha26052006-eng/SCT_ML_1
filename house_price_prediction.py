
# ==========================================================
# SkillCraft Technology - Machine Learning Internship
# Task 01: House Price Prediction using Linear Regression
# ==========================================================

# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 2. Load Dataset

print("\nLoading House Price Dataset...")

data = pd.read_csv("train.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# 3. Select Features and Target Variable

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

X = data[features]
y = data['SalePrice']

print("\nSelected Features:")
print(features)

# 4. Handle Missing Values

imputer = SimpleImputer(strategy='median')

# 5. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 6. Create Linear Regression Model

model = make_pipeline(
    SimpleImputer(strategy='median'),
    LinearRegression()
)

# 7. Train Model

model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# 8. Predict House Prices

predictions = model.predict(X_test)

# 9. Evaluate Model Performance

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")

print("R2 Score:", round(r2, 4))

print("Mean Absolute Error:", round(mae, 2))

print("Mean Squared Error:", round(mse, 2))

print("Root Mean Squared Error:", round(rmse, 2))

# 10. Save Actual and Predicted Results

results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': predictions
})

results.to_csv("prediction_results.csv", index=False)

print("\nFirst 10 Prediction Results:")
print(results.head(10))

# 11. Visualize Actual vs Predicted Prices

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6,
    label="Predicted Prices"
)

# Perfect prediction reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle='--',
    label="Perfect Prediction"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("prediction_graph.png", dpi=300)

plt.show()

# 12. Predict Price of a New House

new_house = pd.DataFrame({
    'GrLivArea': [1710],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

new_prediction = model.predict(new_house)

print("\n========== NEW HOUSE PREDICTION ==========")

print("Living Area: 1710 square feet")
print("Bedrooms: 3")
print("Bathrooms: 2")

print(
    "Predicted House Price:",
    round(new_prediction[0], 2)
)

# 13. Final Status

print("\n==========================================")
print("TASK 01 COMPLETED SUCCESSFULLY!")
print("Linear Regression House Price Prediction")
print("==========================================")
