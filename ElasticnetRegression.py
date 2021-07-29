import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")
X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.preprocessing import PolynomialFeatures
polynomial_converter = PolynomialFeatures(degree=3,include_bias=False)
poly_features = polynomial_converter.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.25, train_size=0.75, random_state=101)

#Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNetCV
elasnet_model = ElasticNetCV(l1_ratio=[.1,.2,.5,.7,.9,.95,.99,1],eps = 0.001, max_iter=1000000)
elasnet_model.fit(X_train, y_train)

elasnet_model.l1_ratio_
test_predictions = elasnet_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,test_predictions)
mse = mean_squared_error(y_test,test_predictions)
rmse = np.sqrt(mse)