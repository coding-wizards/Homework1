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
#entropy 
def entropy(response):
    elements, counts = np.unique(response,return_counts=True)
    entropy=np.sum([(-counts[i]/np.sum(counts))*
                    np.log2(counts[i]/np.sum(counts))for i in 
                    range (len(elements))])
    return entropy

#information gain
    def infogain(data,split_predictor,response="Enjoy"):
        total_entropy=entropy(data[response])
        np.unique(data[split_predictor],return_counts=True)
        Weighted_entropy=np.sum([(counts[i]/np.sum(counts))
        *entropy(data.where(data[split_predictor]==vals[i])
        .dropna()[response])for i in range (len(vals))])
        infogain=total_entropy-Weighted_entropy
        
        return infogain
    
 def ID3(data, originaldata, features, response_name="Enjoy",
         parent_node_class=None):
     if len(np.unique(data[response_name]))<=1:
         return 
     np.unique(data[response_name])[0]
     elif len(data)==0:
     return
         np.unique(originaldata[reponse_name])
         [np.argmax(np.unique(originaldata[response_name],return_counts=True)[1])]
     elif len(features)==0:
     return parent_node_class
 else:
     parent_node_class=np.unique(data[response_name])
     [np.argmax(np.unique(data[response_name],retuen_counts=True)[1])]
     
     item_values=[infogain(data, feature,response_name)for feature in features]
     best_feature_index=np.argmax(item_values)
     best_feature=features[best_feature_index]
     
     tree= {best_feature:{}}
     features=[i for i in features if i != best_feature]
     
     for value in np.unique(data[best_feature]):
         value=value
         
         sub_data=data.where(data[best_feature]==value).dropna()
         subtree=ID3(sub_data,dataset,features,response_name,
                     parent_node_class)
         
         tree[best_feature][value]=subtree
         return(tree)
