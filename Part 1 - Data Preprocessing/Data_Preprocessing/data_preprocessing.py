# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder 

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Taking care of missing Data

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


labelEncoder_x = LabelEncoder();

X[:,0]=labelEncoder_x.fit_transform(X[:,0])


onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
Y = labelEncoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)