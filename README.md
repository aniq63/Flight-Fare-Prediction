# Flight Fare Prediction

This project uses machine learning to predict flight fares based on various factors.  The model is built using Python and several key libraries for data manipulation, visualization, and model training.

## Overview

The project follows these main steps:

1. **Data Loading and Preprocessing:**
    - Loads the flight data from Excel files (Data_Train.xlsx and Test_set.xlsx).
    - Handles missing values by dropping rows with nulls.
    - Extracts relevant features from date and time columns (Date_of_Journey, Dep_Time, Arrival_Time, Duration).  This involves creating new features like Journey_Day, Journey_Month, Journey_Year, Dep_Hour, Dep_Min, Arrival_Hour, Arrival_Min, Duration_Hours, and Duration_Min.
    - Cleans and converts the 'Duration' column to numerical representations of hours and minutes.


2. **Exploratory Data Analysis (EDA):**
   - Visualizes relationships between different features and the target variable ('Price') using seaborn and matplotlib.
   - Analyzes the impact of airlines, destinations, and total stops on flight prices.

3. **Feature Engineering:**
   - Encodes categorical features using one-hot encoding for nominal data (Airline, Source, Destination).
   - Drops irrelevant features like 'Additional_Info' and 'Route'.
   - Applies label encoding to ordinal features (Total_Stops).

4. **Feature Selection:**
   - Uses `ExtraTreesRegressor` to identify the most important features for prediction.

5. **Model Training:**
    - Splits the data into training and testing sets.
    - Scales the numerical features using `StandardScaler`.
    - Trains a `RandomForestRegressor` model on the training data.
    - Performs hyperparameter tuning using `GridSearchCV` to optimize the model's performance.

6. **Model Evaluation:**
    - Evaluates the model's performance on the test set using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

7. **Model Saving:**
    - Saves the trained model using `joblib` for later use.


## Libraries Used

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn (preprocessing, ensemble, model_selection, metrics)
- joblib
- flask
- flask_ngrok (for local development web serving)


## Data Files

The project requires two Excel files:

- `/content/drive/MyDrive/Data_Train.xlsx` (Training Data)
- `/content/drive/MyDrive/Test_set.xlsx` (Test Data)


## Running the Code

1. **Environment setup:** Make sure you have all the necessary libraries installed using the included pip commands.
2. **Data files:**  Place the provided Excel data files in the specified Google Drive location.
3. **Execute the code:** Run the Python script within a Google Colab environment.


## Future Improvements

- Explore different machine learning algorithms to compare performance.
- Further refine feature engineering for improved accuracy.
- Deploy a more robust web application for predictions.

