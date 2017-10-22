# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:15:59 2017

@author: Arafat
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

Data = pd.read_csv('challenge_dataset.csv')

#preparing the train and test data
x_train = Data[['6.1101']]
y_train = Data[['17.592']]

#preparing the model and fitiing data intp the model
model = LinearRegression()
model.fit(x_train, y_train)

#predicting for the known values
y_pred = model.predict(x_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred)
plt.show()
