#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('train.csv', header = 0)
df.info()
print df.describe()
x=[df[(df.Sex=='male')]['Sex'].size,df[(df.Sex=='female')]['Sex'].size]
y=[df[(df.Sex=='male') & (df.Survived == 1)]['Sex'].size,\
   df[(df.Sex=='female') & (df.Survived == 1)]['Sex'].size]
print 'male number: ' + str(x[0]) + '	' + 'female number: ' + str(x[1])
print 'male survive: ' + str(y[0]) + '	' + 'female survive: ' + str(y[1])

df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
df.Cabin[df.Cabin.isnull()]='U0'

age_df = df[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(df.Age.notnull())]
age_df_isnull = age_df.loc[(df.Age.isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rfr.fit(X, Y)
predictAges = rfr.predict(age_df_isnull.values[:,1:])
df.loc[(df.Age.isNull()), 'Age'] = predictAges
