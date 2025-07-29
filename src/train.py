import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# Load the data
df = pd.read_csv("data/bank.csv", sep=';')

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow setup
mlflow.set_experiment("Bank-Marketing-Bagging")

with mlflow.start_run():
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    param_grid = {
        'n_estimators': [10, 20, 30],
        'estimator__max_depth': [3, 5, 10]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log params and metrics
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Save model to disk
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
