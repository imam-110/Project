# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset (replace with your real dataset)
data = {
    'square_footage': [1500, 2000, 1200, 1800, 1350],
    'num_bedrooms': [3, 4, 2, 3, 2],
    'num_bathrooms': [2, 3, 1, 2, 1],
    'neighborhood': ['A', 'B', 'A', 'C', 'B'],
    'price': [200000, 300000, 150000, 250000, 180000]
}

df = pd.DataFrame(data)

# Data Exploration and Visualization
sns.pairplot(df, hue='neighborhood')
plt.show()

# Data Preprocessing
df = pd.get_dummies(df, columns=['neighborhood'])
X = df.drop('price', axis=1)
y = df['price']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualization of Predictions vs. Actual Prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()

# Prediction on New Data (replace with your real data)
new_data = pd.DataFrame({
    'square_footage': [1600],
    'num_bedrooms': [3],
    'num_bathrooms': [2],
    'neighborhood_A': [1],
    'neighborhood_B': [0],
    'neighborhood_C': [0]
})

new_data_scaled = scaler.transform(new_data)
predicted_price = model.predict(new_data_scaled)
print(f'Predicted Price for New Data: {predicted_price[0]}')
# Project
