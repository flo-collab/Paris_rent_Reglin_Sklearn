import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# On charge le dataset
house_data = pd.read_csv("house_data.csv")

# Aperçu du dataset
print(house_data.head())
print(house_data.describe())
print (house_data.info())
house_data = house_data.dropna()
house_data = house_data.astype(int)
print (house_data.info())

X = house_data[house_data.columns[1:]]
y = house_data['price']

print (X.head())

#  on divise notre jeu de données en 2 parties
# 80%, pour l’apprentissage et les 20% restant pour le test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# entrainement  du modèle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model_regLin = LinearRegression()
model_regLin.fit(X_train, y_train)

# on regarde les resultats : Les coefficients
print('Coefficients: \n', model_regLin.coef_)

# Evaluation du training set
from sklearn.metrics import r2_score

y_train_predict = model_regLin.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print ('Le score r² est :',r2,'\nLe score RMSE est : ',rmse)

