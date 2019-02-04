# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:54:04 2019

@author: 1605301
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder 

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values  #This is independent variables matrix
Y = dataset.iloc[:,1].values    #This is dependent var VECTOR

'''
labelEncoder_x = LabelEncoder();

X[:,0]=labelEncoder_x.fit_transform(X[:,0])


onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
Y = labelEncoder_y.fit_transform(Y)
'''
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(X,Y,test_size=1/3, random_state=0)

# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
'''

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set results
# we will create a vector to store predicted

y_pred = regressor.predict(X_test)

# Visualising the the Training set results
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the the Testing set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

 