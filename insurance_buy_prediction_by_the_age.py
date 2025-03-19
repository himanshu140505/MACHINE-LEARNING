# PREDICTING IF A PERSON WOULD BUY A LIFE INSURANCE BASED IN HIS AGE USING LOGISTIC REGRESSION
# Above is a binary logistic regression problem as there are only two possible outcones (i.e. if a person buys insurance or he/she doesnot)

import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

from google_collab import drive
drive.mount('/content/drive')

path = ''

df = pd.read_csv(path) #CSV -> Comma Seperated Values
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.8)

X_train
y_train

from sklearn.linear_model import LogisticRegression
model.fit(X_train, y_train)

model.predice(X_test)

