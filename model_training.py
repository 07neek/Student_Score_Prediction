import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_models(filepath):
    df = pd.read_csv(filepath)

    # Features and Target
    features = ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"]
    target = "Final_Exam_Score"

    X = df[features]
    y = df[target]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Linear Regression ---
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    # --- Polynomial Regression ---
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)

    # --- Model Evaluation ---
    results = {
        "Linear Regression": {
            "R2 Score": r2_score(y_test, y_pred_lin),
            "MSE": mean_squared_error(y_test, y_pred_lin)
        },
        "Polynomial Regression": {
            "R2 Score": r2_score(y_test, y_pred_poly),
            "MSE": mean_squared_error(y_test, y_pred_poly)
        }
    }

    print("\n📊 Model Performance Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name} → R²: {metrics['R2 Score']:.4f}, MSE: {metrics['MSE']:.4f}")

    return lin_reg, poly_reg, poly  # Return trained models and transformer

if __name__ == "__main__":
    train_models("Intership_project\\data\\student_performance_dataset.csv")