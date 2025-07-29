# California_housing_price_prediction

This project uses a machine learning model to predict the **median house value** in California districts based on various housing and geographical features. It includes a data preprocessing pipeline, training with a `RandomForestRegressor`, and support for real-time inference using new input data.

---

## 📌 Problem Statement

Accurately predicting house prices helps in making informed decisions for real estate investments. This project aims to predict the **median house value** using housing and location-related features through a trained machine learning model.

---

## 🚀 Features

- End-to-end data pipeline using `scikit-learn`
- Handles missing values and categorical encoding
- Stratified sampling for balanced train-test split
- Random Forest Regression model for prediction
- Model and pipeline saved for reuse
- Inference support using CSV input or manual user input

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- Joblib

---

## 📁 Project Structure
├── main.py
├── testing.py
├── model.pkl # Saved trained model
├── pipeline.pkl # Saved preprocessing pipeline
├── input.csv # Main Python script
├── output.csv # Output file with predictions
├── housing.csv # Main Python script
├── README.md # Project documentation

---

## 📊 Dataset Description

The dataset contains the following features:

| Feature               | Type        | Description                            |
|-----------------------|-------------|----------------------------------------|
| longitude             | Numerical   | Longitude of the district              |
| latitude              | Numerical   | Latitude of the district               |
| housing_median_age    | Numerical   | Median age of the houses               |
| total_rooms           | Numerical   | Total number of rooms                  |
| total_bedrooms        | Numerical   | Total number of bedrooms               |
| population            | Numerical   | Total population in the district       |
| households            | Numerical   | Number of households                   |
| median_income         | Numerical   | Median income of the residents         |
| ocean_proximity       | Categorical | Distance from the ocean                |
| median_house_value    | Target      | Value to be predicted                  |

---

## ⚙️ How It Works

### 🔧 Training Phase

1. Load the housing dataset from `housing.csv`.
2. Perform **stratified sampling** using income categories.
3. Separate labels (`median_house_value`) from features.
4. Build a preprocessing pipeline:
   - Impute missing values using median strategy
   - Scale numerical features
   - Encode categorical values using one-hot encoding
5. Train a `RandomForestRegressor` model.
6. Save the trained model and pipeline to disk using `joblib`.

### 📈 Inference Phase

1. Load saved model and pipeline (`model.pkl`, `pipeline.pkl`).
2. Read new data from `input.csv`.
3. Transform the input data using the pipeline.
4. Make predictions using the trained model.
5. Save results with predicted values to `output.csv`.

---

