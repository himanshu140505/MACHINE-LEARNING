import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')

# %matplotlib inline 
plt.xlabel('area')
plt.ylabel('prices')
plt.scatter(df.area, df.price, color = 'red', marker = '+')

new_df = df.drop('price', axis = 'columns')
new_df

price = df.price
price

#Create Linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df, price)

reg.predict([[3300]])

reg.coef_

reg.intercept_

# y = mx+c

# Generate CSV file with list of HOME PRICE PREDICTION 

area_df = pd.read_csv('area.csv')
area_df.head(3)

p = reg.predict(area_df)
p

area_df['prices'] = p
area_df

area_df.to_csv("prediction.csv")
