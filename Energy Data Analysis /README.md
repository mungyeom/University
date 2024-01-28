# Project Title: Energy Consumption Analysis

## Overview
This project aims to analyze energy consumption data to predict future energy needs and identify key factors influencing energy prices and consumption patterns. By merging various datasets such as historical electricity and gas prices, client information, and weather conditions, we create a comprehensive dataset for analysis. Machine learning models are then applied to this dataset to forecast energy consumption and understand the impact of different variables.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python (version 3.7 or later)
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Feature-engine

## Setup
1. Clone the repository to your local machine.
2. Ensure that all required Python libraries are installed. You can install them using the command:
    ```
    pip install pandas numpy seaborn matplotlib scikit-learn feature-engine
    ```
3. Place your datasets (`train.csv`, `electricity_prices.csv`, `gas_prices.csv`, `historical_weather.csv`, `weather_station_to_county_mapping.xlsx`, and `client.csv`) in the same directory as the project files.

## Data Preprocessing
Data preprocessing involves cleaning and preparing the data for analysis. This includes handling missing values, merging datasets based on common identifiers, converting data types, and creating new features. The cleaned data is saved to `cleaned_train.csv` and `cleaned_train_final.csv` for further analysis.

## Feature Engineering
We generate cyclical features from date-time columns to capture the cyclical nature of time data (e.g., hours, days, months). This is crucial for models to understand patterns based on time.

## Exploratory Data Analysis (EDA)
EDA is conducted using Seaborn and Matplotlib to understand the distributions of various features and the relationships between them. This step is crucial for gaining insights and guiding the modeling process.

## Model Building
Several regression models, including Linear Regression, Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor, are trained and evaluated. The models aim to predict energy consumption based on the features prepared in the preprocessing stage.

## Model Evaluation
Models are evaluated using metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Cross-validation is used to ensure the models' performance is consistent across different subsets of the dataset.

## Feature Importance Analysis
Analyzing feature importance helps in understanding which features contribute most to the prediction. This is performed for ensemble models like Random Forest and Gradient Boosting.

## Hyperparameter Tuning
Grid Search CV is used to find the optimal hyperparameters for the models, improving their performance.

## Final Model
The final model is selected based on its performance and simplicity. The chosen model is then trained on the entire dataset to make predictions.

## Files and Directories
- `README.md`: This file, containing an overview of the project and instructions for setting up and running the code.
- `data_processing.py`: Python script for data preprocessing and cleaning.
- `feature_engineering.py`: Script for generating new features and preparing the data for modeling.
- `model_training.py`: Contains the code for training and evaluating different machine learning models.
- `utilities.py`: Utility functions used across the project for tasks like data loading and transformation.

## Running the Project
To run the analysis, execute the scripts in the following order:
1. `data_processing.py` for data cleaning and preprocessing.
2. `feature_engineering.py` for generating new features.
3. `model_training.py` for training and evaluating models.

## Contributors
- Mungyeom Kim (MK)

