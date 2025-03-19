import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
data = pd.read_csv('dataset/USA_Housing.csv')

# Data overview
print(data.head())
print(data.info())
print(data.describe())

# Heatmap to show correlation
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()

# Scatter plots
plt.scatter(data['Avg. Area Income'], data['Price'])
plt.show()

# Prepare the data
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
          'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Predicted vs Actual plot
plt.scatter(y_test, predictions)
plt.show()

# Coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)


# Test the model with custom values
test_input = np.array([[75000, 6.5, 7, 4, 40000]])  
predicted_price = model.predict(test_input)
print(f'Predicted House Price for custom input: ${predicted_price[0]:,.2f}')