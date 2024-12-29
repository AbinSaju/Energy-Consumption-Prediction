import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('2.csv')

# Checking dates and set the Date column as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# time-related features
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['DayOfWeek'] = data.index.dayofweek

# Define features (X) and target (y)
X = data[['Year', 'Month', 'Day', 'DayOfWeek']]
y = data['Consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Energy Consumption')
plt.show()

# Forecast future consumption
future_dates = pd.date_range(start='2025-01-01', end='2025-01-10')  # Example future dates
future_features = pd.DataFrame({
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day,
    'DayOfWeek': future_dates.dayofweek
})

future_predictions = model.predict(future_features)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Consumption': future_predictions})
print(forecast_df)












import matplotlib.pyplot as plt

# Ensure y_pred is a pandas Series with the same index as y_test
y_pred = pd.Series(y_pred, index=y_test.index)

# Plot actual vs predicted energy consumption
plt.figure(figsize=(10, 6))

# Plot actual values (blue dashed line)
plt.plot(y_test.index, y_test, label='Actual Consumption', marker='o', linestyle='--', color='blue')

# Plot predicted values (green solid line)
plt.plot(y_test.index, y_pred, label='Predicted Consumption', marker='x', linestyle='-', color='green')

# Add title and labels
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
