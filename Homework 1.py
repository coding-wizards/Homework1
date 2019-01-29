# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:06:51 2019

@author: Varunya Camilo Barrera
"""

import pandas as pd
import numpy as np
from pprint import pprint


#Import the dataset
##f=open("dt-data.txt","r")
##if f.mode=="r":
##    dataset=f.read()
##    print (dataset)

dataset=pd.read_csv('dt-data2.csv',names=['Occupied','Price','Music','Location','VIP','Favorite Beer','Enjoy'])
print(dataset)
dataset=dataset.drop(0,axis=0)
print("test")
print(dataset)