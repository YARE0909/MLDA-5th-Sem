import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import streamlit as st
from io import StringIO
import numpy as np

# Load the CSV file
file_path = 'metadata.csv'  # Update this path to the location of your file if needed
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

# Apply logistic regression
st.header("Logistic Regression: Binary Classification of References Count")

# Create a binary target variable based on a threshold
threshold = df_clean['references-count'].median()
df_clean.loc[:, 'references_class'] = (df_clean['references-count'] > threshold).astype(int)

# Define features and target for logistic regression
X_logistic = df_clean[['issued']]
y_logistic = df_clean['references_class']

# Split the data into training and testing sets for logistic regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42
)

# Create and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)

# Make predictions and evaluate performance for logistic regression
y_pred_logistic = logistic_model.predict(X_test_logistic)
accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)
precision_logistic = precision_score(y_test_logistic, y_pred_logistic, zero_division=1)
recall_logistic = recall_score(y_test_logistic, y_pred_logistic)
f1_logistic = f1_score(y_test_logistic, y_pred_logistic)

st.header("Logistic Regression Performance")
st.write(f"Accuracy: {accuracy_logistic:.2f}")
st.write(f"Precision: {precision_logistic:.2f}")
st.write(f"Recall: {recall_logistic:.2f}")
st.write(f"F1 Score: {f1_logistic:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test_logistic, y_pred_logistic)

st.write("Confusion Matrix:")
st.write(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Below Threshold', 'Above Threshold'])
plt.yticks(tick_marks, ['Below Threshold', 'Above Threshold'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
st.pyplot(plt)

# Plotting Actual vs Predicted Classes
st.header("Logistic Regression: Actual vs Predicted Classes")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_logistic, y_test_logistic, color='green', label='Actual Class')
plt.scatter(X_test_logistic, y_pred_logistic, color='red', marker='x', label='Predicted Class')
plt.xlabel('Year of Publication')
plt.ylabel('Class (0: Below Threshold, 1: Above Threshold)')
plt.title('Logistic Regression: Actual vs Predicted Classes')
plt.legend()
st.pyplot(plt)

# Decision boundary
st.header("Decision Boundary")

# Predict the probability for the test set
probabilities_test = logistic_model.predict_proba(X_test_logistic)[:, 1]

# Calculate decision boundary based on model coefficients
decision_boundary = -logistic_model.intercept_[0] / logistic_model.coef_[0][0]
x_vals = np.array(plt.gca().get_xlim())
y_vals = -(x_vals * logistic_model.coef_[0][0] + logistic_model.intercept_[0]) / logistic_model.coef_[0][0]

plt.figure(figsize=(10, 6))
plt.scatter(X_test_logistic, probabilities_test, c=y_test_logistic, cmap='viridis', edgecolors='k', alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, '--', color='red', label='Decision Boundary')
plt.xlabel('Year of Publication')
plt.ylabel('Predicted Probability')
plt.title('Logistic Regression: Decision Boundary')
plt.legend()
st.pyplot(plt)
