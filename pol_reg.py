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

train_rmse_error = []
test_rmse_error = []

for d in range(1,10):
    poly_converter = PolynomialFeatures(degree = d, include_bias = False)
    pol_features = poly_converter.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(pol_features, y, train_size=0.75, test_size=0.25, random_state=101)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_absolute_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_absolute_error(y_test, pred_test))
    
    train_rmse_error.append(train_rmse)
    test_rmse_error.append(test_rmse)