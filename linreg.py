import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Advertising.csv')
data.head()
X = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, train_size=0.75, random_state=101)
len(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

test_predictions = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score

#MAE
mean_absolute_error(y_test, test_predictions)

#RMSE
np.sqrt(mean_squared_error(y_test, test_predictions))

test_residuals = y_test - test_predictions
sns.scatterplot(x = y_test, y=test_residuals)
plt.axhline(y=0, color = 'r', )

sns.displot(test_residuals, bins = 30, kde = True)


final_model = LinearRegression()
final_model.fit(X,y)
final_model.coef_


from joblib import dump, load
dump(final_model, 'final_model.joblib')

load_model = load('final_model.joblib')
load_model
