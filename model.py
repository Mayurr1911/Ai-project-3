import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Generate synthetic traffic data
np.random.seed(42)
n_samples = 1000
hour = np.random.randint(0, 24, n_samples)
day = np.random.randint(0, 7, n_samples)
weather = np.random.uniform(0, 1, n_samples)
traffic_volume = hour * 10 + day * 5 + weather * 20 + np.random.normal(0, 10, n_samples)

data = pd.DataFrame({'hour': hour, 'day': day, 'weather': weather, 'traffic_volume': traffic_volume})

X = data[['hour', 'day', 'weather']]
y = data['traffic_volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(model, 'traffic_prediction_model.pkl')
