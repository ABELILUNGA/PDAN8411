
# Abel ilunga st10090262

#  Part 1: Evaluate Dataset Suitability
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("insurance.csv")

# Basic info and checks
print("Dataset Info:\n")
df.info()
print("\nSummary Statistics:\n")
print(df.describe(include='all'))
print("\nFirst 5 Records:\n")
print(df.head())

# - Observation:
# - No missing values
# - Target variable ('charges') is continuous
# - Suitable for Linear Regression

#  Part 2: Plan the Analysis
# Documented separately  Report

#  Part 3: Conduct EDA

# Visualizing the distribution of numerical features to check for skewness, spread, and potential anomalies
numerical_cols = ['age', 'bmi', 'children', 'charges']
df[numerical_cols].hist(bins=30, figsize=(10, 8))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Creating boxplots to identify outliers and understand the distribution of each numerical variable
for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Scatter plots to explore relationships between features (age, BMI, children) and charges
# Color-coded by 'smoker' to highlight the impact of smoking on medical costs
for col in ['age', 'bmi', 'children']:
    plt.figure()
    sns.scatterplot(x=df[col], y=df['charges'], hue=df['smoker'])
    plt.title(f"{col} vs Charges")
    plt.show()

#  Encode categorical variables
# Convert categorical columns (e.g., sex, smoker, region) into binary features using one-hot encoding
# This is required for linear regression to process non-numeric data
df_encoded = pd.get_dummies(df, drop_first=True)

#  Correlation heatmap
# Displays pairwise correlations between features to identify strong relationships or multicollinearity
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



#  OLS Regression for feature significance
# Using statsmodels to calculate p-values and check which features significantly affect the target variable
import statsmodels.api as sm

X = df_encoded.drop("charges", axis=1)  # Features (excluding the target)
y = df_encoded["charges"]              # Target variable
X_const = sm.add_constant(X)           # Add intercept to the model

X_const = X_const.astype(float)
y = y.astype(float)

# Fit the OLS model and print a detailed statistical summary (includes p-values, R-squared, etc.)
ols_model = sm.OLS(y, X_const).fit()
print(ols_model.summary())

#  Part 3c: Train the Model

# Splitting the dataset into training and testing sets (80% train, 20% test)
# This ensures the model is evaluated on unseen data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training a Linear Regression model using the training set
# This will learn the coefficients for each input feature
lr = LinearRegression()
lr.fit(X_train, y_train)


#  Part 4: Evaluate the Model

# Import evaluation metrics to assess how well our model performs
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict target variable (medical charges) using the test dataset
y_pred = lr.predict(X_test)

# R² score tells us how much of the variance in the target is explained by the model (higher is better, max = 1)
r2 = r2_score(y_test, y_pred)

# RMSE (Root Mean Squared Error) gives us the average prediction error in the same units as 'charges'
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE (Mean Absolute Error) is the average absolute difference between predictions and actual values
mae = mean_absolute_error(y_test, y_pred)

# Print the results to evaluate model performance
print("\nModel Performance:")
print(f"R² Score: {r2:.3f}")     # Proportion of variance explained
print(f"RMSE: {rmse:.2f}")       # How far predictions deviate, on average
print(f"MAE: {mae:.2f}")         # Average size of prediction errors


#  Part 4b: Retrain (optional step based on performance)
# Example: Add interaction term (if needed)
# df_encoded['bmi_smoker'] = df_encoded['bmi'] * df_encoded['smoker_yes']
# Retrain with new feature

#  Part 5: Report will be generated separately