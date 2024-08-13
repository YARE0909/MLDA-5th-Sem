import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from io import StringIO
import numpy as np

# Load the CSV file
file_path = 'metadata.csv'  # Update this path to the location of your file
df = pd.read_csv(file_path)

st.title("Data Analysis and Linear Regression")

# Display the first few rows of the dataframe
st.header("First Few Rows of the Dataframe")
st.dataframe(df.head())

# Check the shape of the data
st.header("Shape of the Dataframe")
st.write(df.shape)

# Get basic information about the data
st.header("Basic Information about the Dataframe")
buffer = StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
st.text(info)

# Get summary statistics
st.header("Summary Statistics of the Dataframe")
st.write(df.describe())

# Display the column names
st.header("Column Names of the Dataframe")
st.write(df.columns.tolist())

# Convert 'issued' column to datetime format and extract the year
df['issued'] = pd.to_datetime(df['issued'], errors='coerce').dt.year

# Drop rows with NaT values in 'issued'
df = df.dropna(subset=['issued'])

# Plot the distribution of the 'issued' column
st.header("Distribution of Articles by Year")
plt.figure()
plt.hist(df['issued'], bins=30)
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.title(f'Distribution of Articles by Year')
st.pyplot(plt)

# Plot the distribution of the 'references-count' column
st.header("Distribution of References-Count with Potential Outliers")
plt.figure(figsize=(10, 6))
plt.hist(df['references-count'], bins=100, edgecolor='k', alpha=0.7)
plt.xlabel('References-Count')
plt.ylabel('Frequency')
plt.title('Distribution of References-Count with Potential Outliers')

# Identify and highlight potential outliers
q1 = df['references-count'].quantile(0.25)
q3 = df['references-count'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['references-count'] < lower_bound) | (df['references-count'] > upper_bound)]

# Highlight outliers on the histogram
plt.scatter(outliers['references-count'], [0] * len(outliers), color='red', label='Outliers')
plt.legend()
st.pyplot(plt)

st.write(f"Number of outliers in 'references-count': {len(outliers)}")
st.write("Outliers:")
st.dataframe(outliers)

# Drop the outliers for linear regression
df_clean = df[(df['references-count'] >= lower_bound) & (df['references-count'] <= upper_bound)]

# Define the features and target variable
X_clean = df_clean[['issued']]
y_clean = df_clean['references-count']

# Split the dataset into training and testing sets for clean data
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model with clean data
model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train_clean)

# Make predictions and evaluate performance on clean data
y_pred_clean = model_clean.predict(X_test_clean)
mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
rmse_clean = np.sqrt(mse_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

# Plot the regression line for clean data
st.header("Linear Regression: Year vs. References Count (Clean Data)")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_clean, y_test_clean, color='blue', label='Actual')
plt.plot(X_test_clean, y_pred_clean, color='red', linewidth=2, label='Predicted')
plt.xlabel('Year of Publication')
plt.ylabel('References Count')
plt.title('Linear Regression: Year vs. References Count (Clean Data)')
plt.legend()
st.pyplot(plt)

st.header("Linear Regression Performance (Clean Data)")
st.write(f"Mean Squared Error: {mse_clean:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_clean:.2f}")
st.write(f"R^2 Score: {r2_clean:.2f}")

# Add noise to the data
np.random.seed(42)
noise = np.random.normal(0, 1, size=y_clean.shape)
y_noisy = y_clean + noise

# Define the features and target variable with noise
X_noisy = X_clean.copy()
y_noisy = pd.Series(y_noisy, name='references-count')

# Split the dataset into training and testing sets for noisy data
X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model with noisy data
model_noisy = LinearRegression()
model_noisy.fit(X_train_noisy, y_train_noisy)

# Make predictions and evaluate performance on noisy data
y_pred_noisy = model_noisy.predict(X_test_noisy)
mse_noisy = mean_squared_error(y_test_noisy, y_pred_noisy)
rmse_noisy = np.sqrt(mse_noisy)
r2_noisy = r2_score(y_test_noisy, y_pred_noisy)

# Plot the regression line for noisy data
st.header("Linear Regression: Year vs. References Count (Noisy Data)")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_noisy, y_test_noisy, color='green', label='Actual')
plt.plot(X_test_noisy, y_pred_noisy, color='orange', linewidth=2, label='Predicted')
plt.xlabel('Year of Publication')
plt.ylabel('References Count')
plt.title('Linear Regression: Year vs. References Count (Noisy Data)')
plt.legend()
st.pyplot(plt)

st.header("Linear Regression Performance (Noisy Data)")
st.write(f"Mean Squared Error: {mse_noisy:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_noisy:.2f}")
st.write(f"R^2 Score: {r2_noisy:.2f}")

st.header("Linear Regression Performance (Clean Data)")
st.write(f"Mean Squared Error: {mse_clean:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_clean:.2f}")
st.write(f"R^2 Score: {r2_clean:.2f}")

# Cross-validation on clean data
st.header("Cross-Validation Performance (Clean Data)")
cv_scores_clean = cross_val_score(model_clean, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error')
mean_cv_score_clean = -cv_scores_clean.mean()
st.write(f"Mean Cross-Validated Mean Squared Error: {mean_cv_score_clean:.2f}")

# Cross-validation on noisy data
st.header("Cross-Validation Performance (Noisy Data)")
cv_scores_noisy = cross_val_score(model_noisy, X_noisy, y_noisy, cv=5, scoring='neg_mean_squared_error')
mean_cv_score_noisy = -cv_scores_noisy.mean()
st.write(f"Mean Cross-Validated Mean Squared Error: {mean_cv_score_noisy:.2f}")
