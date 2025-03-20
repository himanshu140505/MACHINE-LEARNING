import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('DATASETS/homeprices.csv')

# Check dataset structure
print(df.head())

# Plot data
plt.xlabel('Area')
plt.ylabel('Prices')
plt.scatter(df['area'], df['price'], color='red', marker='+')

# Prepare data for training
X = df[['area']]  # Independent variable
y = df['price']   # Dependent variable

# Create and train linear regression model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Predict price for 3300 sq. ft.
predicted_price = reg.predict(pd.DataFrame([[3300]], columns=['area'])) 
print(f"\nPredicted price for 3300 sq. ft: {predicted_price[0]}")

# Print coefficients
print(f"\nCoefficient (m): {reg.coef_[0]}")
print(f"\nIntercept (c): {reg.intercept_}")

# Load new area values
area_df = pd.read_csv('DATASETS/areas.csv')

# Predict prices
p = reg.predict(area_df[['area']])  # Ensure correct column selection
area_df['prices'] = p

# Save predictions
area_df.to_csv("PREDICTIONS/lr_prediction.csv", index=False)

print("\nPredictions saved to prediction.csv")
