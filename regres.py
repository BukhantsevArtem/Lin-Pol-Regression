import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Advertising.csv')
data.head()
data['Total'] = data['TV']+ data['radio']+data['newspaper']

plt.figure(dpi = 200)
sns.regplot(x='Total', y='sales', data = data)

X = data['Total']
y = data['sales']

np.polyfit(X,y, deg = 1)

potent = np.linspace(0,500,100)
predicted = potent*0.04868788+4.24302822

plt.plot(potent, predicted)
