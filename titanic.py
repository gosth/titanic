#!/usr/bin/env python
# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

np.set_printoptions(precision = 4, threshold = 10000, linewidth = 160, edgeitems = 999, suppress = True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 160)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 4)

def processCabin():
    global df
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x : getCabinLetter(x))
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    
    if keep_binary:
        cletters = pd.get_dummies(df['CabinLetter']).rename(columns = lambda x: 'CabinLetter_' + str(x))
        df = pd.concat([df, cletters], axis =1)
    df['CabinNumber'] = df['Cabin'].map(lambda x : getCabinNumber(x)).astype(int) + 1
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'])
    
        
def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 'U'
    
def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0
        
    

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

'''
age_df = df[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(df.Age.notnull())]
age_df_isnull = age_df.loc[(df.Age.isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rfr.fit(X, Y)
predictAges = rfr.predict(age_df_isnull.values[:,1:])
df.loc[(df.Age.isNull()), 'Age'] = predictAges
'''

def getDataSets(binary = False, bins = False, scaled = False, strings = False, \
                raw = True, pca = False, balanced = False):
    global keep_binary, keep_bins, keep_scaled, keep_raw, keep_strings, df
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_raw = raw
    keep_strings = strings
    
    input_df = pd.read_csv('train.csv', header = 0)
    submit_df = pd.read_csv('test.csv', header = 0)
    df = pd.concat([input_df, submit_df])
    df.reset_index(inplace = True)
    df.drop('index', axis = 1, inplace = True)
    df = df.reindex_axis(input_df.columns, axis = 1)
    processCabin()
    print df
    
    
    
    
    
    
    
    

if __name__ == '__main__':
        train, test = getDataSets(bins = True, scaled = True, binary = True)
        #drop_list = [

