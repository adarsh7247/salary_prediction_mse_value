# salary_prediction_mse_value

Step 1: Import Necessary Libraries

- Import the required libraries:
    - pandas for data manipulation and analysis
    - numpy for numerical operations
    - train_test_split from scikit-learn for splitting data into training and testing sets
    - LinearRegression from scikit-learn for creating a linear regression model
    - mean_squared_error from scikit-learn for evaluating model performance

Code:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


Step 2: Create a Sample Dataset

- Define a dictionary data with YearsExperience and Salary as keys
- Convert the dictionary into a Pandas DataFrame df

Code:

data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [40000, 42000, 45000, 48000, 50000, 55000, 60000, 62000, 65000, 70000]
}
df = pd.DataFrame(data)


Step 3: Define Features and Target Variable

- Define the feature (independent variable) as YearsExperience
- Define the target variable (dependent variable) as Salary

Code:

X = df[['YearsExperience']]
y = df['Salary']


Step 4: Split the Data into Training and Testing Sets

- Split the data into 80% training data and 20% testing data using train_test_split

Code:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Step 5: Create and Fit the Linear Regression Model

- Initialize a LinearRegression model
- Fit the model on the training data using fit()

Code:

model = LinearRegression()
model.fit(X_train, y_train)


Step 6: Make Predictions on the Test Set

- Predict salary values for the test set using predict()

Code:

y_pred = model.predict(X_test)


Step 7: Calculate Mean Squared Error

- Calculate the mean squared error of the model's predictions using mean_squared_error()

Code:

mse = mean_squared_error(y_test, y_pred)


Step 8: Print the Results

- Print the mean squared error value

Code:

print("Mean Squared Error:", mse)

