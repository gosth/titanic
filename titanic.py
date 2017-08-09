#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv('train.csv', header = 0)
df.info()
print df.describe()
x=[df[(df.Sex=='male')]['Sex'].size,df[(df.Sex=='female')]['Sex'].size]
y=[df[(df.Sex=='male') & (df.Survived == 1)]['Sex'].size,\
   df[(df.Sex=='female') & (df.Survived == 1)]['Sex'].size]
print 'male number: ' + str(x[0]) + '	' + 'female number: ' + str(x[1])
print 'male survive: ' + str(y[0]) + '	' + 'female survive: ' + str(y[1])
