import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from fastapi import FastAPI
import streamlit as st

# Load dataset (Replace with actual dataset path)
data = pd.read_csv('fraud_data.csv')

# Explore data
print(data.head())
print(data.info())

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Feature selection and preprocessing
X = data.drop(columns=['is_fraud'])  # Assuming 'is_fraud' is the target variable
y = data['is_fraud']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training - Comparing multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_accuracy = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Best Model Selected:", best_model)

# Save the best model
joblib.dump(best_model, "fraud_model.pkl")

# Deep Learning Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the neural network model
nn_model.save("fraud_nn_model.h5")

# API Setup
app = FastAPI()

@app.post("/predict")
def predict_fraud(data: dict):
    model = joblib.load("fraud_model.pkl")
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return {"fraud": bool(prediction[0])}

# Streamlit Dashboard
st.title("Fraud Detection System")
file = st.file_uploader("Upload a CSV file for prediction", type=['csv'])
if file:
    df = pd.read_csv(file)
    model = joblib.load("fraud_model.pkl")
    predictions = model.predict(df)
    df['Fraud Prediction'] = predictions
    st.write(df)
    st.download_button("Download Predictions", df.to_csv(index=False), "text/csv")
