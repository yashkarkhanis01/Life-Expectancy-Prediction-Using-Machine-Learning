import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('life_expectancy_data.csv')

# Strip whitespace from the column names
df.columns = df.columns.str.strip()

# Display the column names to verify
print("Columns in the dataset:", df.columns)

# Overview of the dataset
print(df.info())
print(df.describe())

# Fill NaN values with the mean of the respective columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Ensure non-numeric columns are not used in the correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Verify if 'Life_expectancy' column exists and its correct name
print("Columns after NaN filling:", df.columns)

# Pairplot
sns.pairplot(df, x_vars=['GDP', 'Alcohol', 'BMI'], y_vars='Life expectancy', height=4, aspect=1, kind='scatter')
plt.suptitle('Pairplot of GDP, Alcohol, BMI vs Life Expectancy', y=1.02)
plt.show()

# Feature selection
features = ['GDP', 'Alcohol', 'BMI', 'Adult Mortality', 'Hepatitis B', 'Measles', 'Polio', 'Total expenditure']
X = df[features]
y = df['Life expectancy']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Life Expectancy')
plt.show()
