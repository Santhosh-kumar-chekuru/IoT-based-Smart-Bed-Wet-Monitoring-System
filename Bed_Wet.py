from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Prepare features (X) and labels (y)
X = data[['moisture_level', 'movement_data']]
y = data['wetness_label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
print("Model Coefficients:", model.coef_)
