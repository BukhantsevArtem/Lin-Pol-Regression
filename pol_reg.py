import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Advertising.csv')
X = data.drop('sales',axis=1)
y = data['sales']

from sklearn.preprocessing import PolynomialFeatures

polynom = PolynomialFeatures(degree = 2, include_bias=False)
pol_features = polynom.fit_transform(X)
pol_features[0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pol_features, y, train_size=0.75, test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_test, predictions)
mean_squared_error(y_test, predictions)