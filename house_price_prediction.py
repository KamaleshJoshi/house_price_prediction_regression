# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
# Make sure the dataset is in the correct path
data = pd.read_csv('dataset/USA_Housing.csv')

# Data overview
print("\n First 5 rows of the dataset:")
print(data.head())  # First five rows of the dataset

print("\n Dataset Info:")
print(data.info())  # Structure and data types

print("\n Statistical Summary:")
print(data.describe())  # Summary statistics like mean, std, min, max

# Drop irrelevant columns (if any)
if 'Address' in data.columns:
    data.drop(['Address'], axis=1, inplace=True)

# Heatmap to show correlation
# Compute the correlation and plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Scatter plot with regression line (Avg. Area Income vs Price)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Avg. Area Income'], y=data['Price'], color='blue', label='Actual Values')

# Fit the regression line using numpy's polyfit (degree 1 = straight line)
m, b = np.polyfit(data['Avg. Area Income'], data['Price'], 1)
plt.plot(data['Avg. Area Income'], m * data['Avg. Area Income'] + b, color='red', label='Regression Line')

# Add labels and legend
plt.title('Avg. Area Income vs Price')
plt.xlabel('Avg. Area Income')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prepare the data
# Define independent and dependent variables
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
          'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

# Split into training and test sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Train the model
# Create a LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)  # Fit the model on training data

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("\n Model Evaluation:")
print('MAE:', metrics.mean_absolute_error(y_test, predictions))   # Mean Absolute Error
print('MSE:', metrics.mean_squared_error(y_test, predictions))     # Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Root Mean Squared Error

# Predicted vs Actual plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, color='blue', label='Predictions')

# Add a reference line (perfect prediction line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Fit')

# Add labels and legend
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()

# Coefficients (Impact of each feature)
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n Model Coefficients:")
print(coeff_df)

# Test the model with custom values
# Create a sample input to test the model
test_input = np.array([[75000, 6.5, 7, 4, 40000]])

# Ensure input shape matches model expectations
try:
    predicted_price = model.predict(test_input)
    print(f'\n Predicted House Price for custom input: ${predicted_price[0]:,.2f}')
except Exception as e:
    print(f"Error in prediction: {e}")

# Save the model (optional)
import joblib
joblib.dump(model, 'house_price_model.pkl')
print("\n Model saved as 'house_price_model.pkl'")

