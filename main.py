import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import numpy as np
from io import StringIO
from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# Assuming the uploaded file is 'metadata.csv'
df = pd.read_csv(next(iter(uploaded.keys())))

# Display the first few rows of the dataframe
print("First Few Rows of the Dataframe:")
print(df.head())

# Check the shape of the data
print("\nShape of the Dataframe:")
print(df.shape)

# Get basic information about the data
print("\nBasic Information about the Dataframe:")
buffer = StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
print(info)

# Get summary statistics
print("\nSummary Statistics of the Dataframe:")
print(df.describe())

# Display the column names
print("\nColumn Names of the Dataframe:")
print(df.columns.tolist())

# Convert 'issued' column to datetime format and extract the year
df['issued'] = pd.to_datetime(df['issued'], errors='coerce').dt.year

# Drop rows with NaT values in 'issued'
df = df.dropna(subset=['issued'])

# Plot the distribution of the 'issued' column
print("\nDistribution of Articles by Year:")
plt.figure()
plt.hist(df['issued'], bins=30)
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.title(f'Distribution of Articles by Year')
plt.show()

# Plot the distribution of the 'references-count' column
print("\nDistribution of References-Count with Potential Outliers:")
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
plt.show()

print(f"Number of outliers in 'references-count': {len(outliers)}")
print("Outliers:")
print(outliers)

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
print("\nLinear Regression: Year vs. References Count (Clean Data)")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_clean, y_test_clean, color='blue', label='Actual')
plt.plot(X_test_clean, y_pred_clean, color='red', linewidth=2, label='Predicted')
plt.xlabel('Year of Publication')
plt.ylabel('References Count')
plt.title('Linear Regression: Year vs. References Count (Clean Data)')
plt.legend()
plt.show()

print("\nLinear Regression Performance (Clean Data)")
print(f"Mean Squared Error: {mse_clean:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_clean:.2f}")
print(f"R^2 Score: {r2_clean:.2f}")

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
print("\nLinear Regression: Year vs. References Count (Noisy Data)")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_noisy, y_test_noisy, color='green', label='Actual')
plt.plot(X_test_noisy, y_pred_noisy, color='orange', linewidth=2, label='Predicted')
plt.xlabel('Year of Publication')
plt.ylabel('References Count')
plt.title('Linear Regression: Year vs. References Count (Noisy Data)')
plt.legend()
plt.show()

print("\nLinear Regression Performance (Noisy Data)")
print(f"Mean Squared Error: {mse_noisy:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_noisy:.2f}")
print(f"R^2 Score: {r2_noisy:.2f}")

# Cross-validation on clean data
print("\nCross-Validation Performance (Clean Data)")
cv_scores_clean = cross_val_score(model_clean, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error')
mean_cv_score_clean = -cv_scores_clean.mean()
print(f"Mean Cross-Validated Mean Squared Error: {mean_cv_score_clean:.2f}")

# Cross-validation on noisy data
print("\nCross-Validation Performance (Noisy Data)")
cv_scores_noisy = cross_val_score(model_noisy, X_noisy, y_noisy, cv=5, scoring='neg_mean_squared_error')
mean_cv_score_noisy = -cv_scores_noisy.mean()
print(f"Mean Cross-Validated Mean Squared Error: {mean_cv_score_noisy:.2f}")

# Apply logistic regression
print("\nLogistic Regression: Binary Classification of References Count")

# Create a binary target variable based on a threshold
threshold = df_clean['references-count'].median()
df_clean = df_clean.copy()  # Create a deep copy to avoid the SettingWithCopyWarning
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

print("\nLogistic Regression Performance")
print(f"Accuracy: {accuracy_logistic:.2f}")
print(f"Precision: {precision_logistic:.2f}")
print(f"Recall: {recall_logistic:.2f}")
print(f"F1 Score: {f1_logistic:.2f}")

# Plotting the logistic regression results
probabilities_test = logistic_model.predict_proba(X_test_logistic)[:, 1]
decision_boundary = -logistic_model.intercept_[0] / logistic_model.coef_[0][0]
x_vals = np.linspace(X_test_logistic.min(), X_test_logistic.max(), 100)
y_vals = -(x_vals * logistic_model.coef_[0][0] + logistic_model.intercept_[0]) / logistic_model.coef_[0][0]

plt.figure(figsize=(10, 6))
plt.scatter(X_test_logistic, probabilities_test, c=y_test_logistic, cmap='viridis', edgecolors='k', alpha=0.7, label='Data Points')
plt.plot(x_vals, y_vals, '--', color='red', label='Decision Boundary')
plt.xlabel('Year of Publication')
plt.ylabel('Predicted Probability')
plt.title('Logistic Regression: Decision Boundary')
plt.legend()
plt.colorbar(label='Predicted Probability')
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test_logistic, y_pred_logistic)

print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
plt.matshow(conf_matrix, cmap='Blues', fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
