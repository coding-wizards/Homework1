#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:06:51 2019

@author: camilo barrera, varunya ilanghovan
"""

import pandas as pd
import numpy as np
from pprint import pprint 

###Import the dataset###
dataset=pd.read_csv('dt-data2.csv',names=['Occupied','Price','Music','Location','VIP','Favorite Beer','Enjoy',])
#print(dataset)


###Function to compute the Entropy###
def entropy(attribute):
    elements, counts=np.unique(attribute, return_counts=True)
    entropy=np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

###Function to compute the information gained###
def InformationGain(data,attribute,response="Enjoy"):
    ##Compute the entropy of the entire dataset
    total_entropy=entropy(data[response])
    
    #Compute the vlaues and counts of the splitting attribute
    vals,counts=np.unique(data[attribute],return_counts=True)
    
    #Compute the weighted entropy
    w_entropy=np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[attribute]==vals[i]).dropna()[response]) for i in range(len(vals))])
    
    #Compute the information gained
    info_gain=total_entropy-w_entropy
    
    return info_gain

###Function to Create the trees (ID3 algorithm)
def tree_ID3(data,dataset,features,response="Enjoy",parent_node=None):
    #Define stopping criteria
    ##If all responses have the same value, return the value
    if len(np.unique(data[response]))<=1:
        return np.unique(data[response])[0]
    
    ##If the dataset is empty
    elif len(data)==0:
        return np.unique(dataset[response])[np.argmax(np.unique(dataset[response],return_counts=True)[1])]
        
    #If the feature space is empty, return the mode target feature value of the direct parent node
    elif len(features)==0:
        return parent_node
        
    #If none of the above holds, grow the tree
    else:
        parent_node=np.unique(data[response])[np.argmax(np.unique(data[response],return_counts=True)[1])]

        #Select the fueature that best splits the dataset
        ##the function argmax ensures that in case of a tie, the first occurrence (the feature that is closer to the front of the list) is chosen as the best feature
        item_values=[InformationGain(data,feature,response) for feature in features]
        best_feature_index=np.argmax(item_values)
        best_feature=features[best_feature_index]
    
        #Creathe the tree structure
        tree={best_feature:{}}
    
        #Remove the feature with the best information
        features=[i for i in features if i != best_feature]
            
        #Grow a branch under the root node for each possible value of the root feature node
        for value in np.unique(data[best_feature]):
            value=value
    
            #Split the datset along the value of the feature with the largest information gain and create subsets
            sub_data=data.where(data[best_feature]==value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters
            subtree=tree_ID3(sub_data,dataset,features,response,parent_node)
            
            #Add the subree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value]=subtree
        return(tree)
            
###Function to predict new querys
def predict(query, tree, default='Yes'):
    for key in list(query.keys()):
        try:
            result=tree[key][query[key]]
        except:
            return default
        result=tree[key][query[key]]
        if isinstance(result,dict):
            return predict(query,result)
        else:
            return result
        
tree=tree_ID3(dataset,dataset,dataset.columns[:-1])
pprint(tree)
new_val={"Occupied":'Moderate',"Price":"Cheap","Music":"Loud","Location":"City-Center ","VIP":"No","Favorite Beer":"No"}
print("new_val: ",new_val)
prediction=predict(new_val,tree)
print("prediction: ",prediction)



#using sklearn

from sklearn.tree import DecisionTreeClassifier
train_features=dataset.iloc[1:11,:6]
test_features=dataset.iloc[11:,:6]
train_targets=dataset.iloc[1:11,-1]
test_targets=dataset.iloc[11:,-1]

train_x1=pd.get_dummies(train_features)
train_y1=pd.get_dummies(train_targets)
test_x=pd.get_dummies(test_features)
test_y=pd.get_dummies(test_targets)

tree= DecisionTreeClassifier(criterion='entropy').fit(train_x1,train_y1)
prediction=tree.predict(test_x)
print(tree.score(test_x,test_y)*100,"%")
