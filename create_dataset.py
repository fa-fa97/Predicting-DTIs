#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:49:34 2019

@author: fatemehf
"""

import pandas as pd 


pcols = ['Source ID', 'Target ID', 'Edge Type','Edge ID']
ncols = ['Source ID', 'Target ID', 'Edge Type']

neg_data = pd.read_csv('~/Desktop/final_project/DTINet/data/generated_data/neg_data.csv', sep=' ',names=ncols)
pos_data = pd.read_csv('~/Desktop/final_project/DTINet/data/generated_data/main_db.csv', sep=' ',names=pcols)

neg_data['Label'] = 0
pos_data['Label'] = 1

column = ['F'+str(i) for i in range(1,129)]

column.insert(0,'Node ID')

embedding=pd.read_csv('~/Desktop/final_project/edge2vec/vector_db.txt',sep=' ', names=column)

for i in range(1,129):
    embedding['F'+str(i)] = embedding['F'+str(i)].apply(lambda x : float(x))
   
embedding['Node ID'] = embedding['Node ID'].apply(lambda x : float(x))

cols = ['F'+str(i) for i in range(1,129)]    
cols.insert(0,'Source ID')
cols.insert(1,'Target ID')
cols.insert(2,'Label')

dataset = pd.DataFrame(columns=cols,data=None) 

for i,row in neg_data.iterrows():
    label=row['Label']
    dataset_full = []
    target_vector = embedding.loc[embedding['Node ID'] == row['Target ID']]
    target_vector = target_vector.reset_index(drop=True)
    source_vector = embedding.loc[embedding['Node ID'] == row['Source ID']]
    source_vector = source_vector.reset_index(drop=True)
    
    dataset_vector = source_vector.subtract(target_vector)
    
    dataset_vector= dataset_vector.drop(columns = ['Node ID'])
    dataset_vector['Label'] = label
    dataset_vector['Source ID'] = source_vector['Node ID']
    dataset_vector['Target ID'] = target_vector['Node ID']
    
    dataset_full.append(list(dataset_vector))
    print(i)


dataset.to_csv('~/Desktop/DTINet/data/generated_data/dataset_neg.csv', sep=' ',index=False)  
