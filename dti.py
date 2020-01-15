#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:45:00 2019

@author: fatemehf
"""

import pandas as pd 
  
#Takes the adjacency matrix m*m on file input_filename into a list of edges
def adj_to_list_identical(input_filename,delimiter,x3,x4,edge_type):
    
    A=pd.read_csv(input_filename,header=None,delimiter=delimiter)
    List=[]
    for source in range(1,A.shape[1]):
        for target in range(0,A.shape[0]-1):
            if A[source][target] == 1:
                List.append((source+x3,target+x4,edge_type))
    return List
    
    
#Takes the adjacency matrix m*n on file input_filename into a list of edges
def adj_to_list_noidentical(input_filename,delimiter,x3,x4,edge_type):
    
    A=pd.read_csv(input_filename,header=None,delimiter=delimiter)
    List=[]
  
    for source in range(0,A.shape[1]-1):
        for target in range(0,A.shape[0]):
            if A[source][target] == 1:
                List.append((target+x4,source+x3,edge_type))
    return List
    
def create_dataframe(List):
    
    dataframe = pd.DataFrame(data=List,columns=['Source ID', 'Target ID', 'Edge Type'])
    return dataframe
    
def print_length(input_df,string):
    
    length = len(input_df)
    print(str(string)+":",length)
    
def main(): 
    
    #extract relations
        
    L_drug_drug = adj_to_list_identical('~/Desktop/final_project/DTINet/data/mat_drug_drug.txt',' ',0,0,1)    
    L_protein_protein = adj_to_list_identical('~/Desktop/final_project/DTINet/data/mat_protein_protein.txt',' ',707,707,2)
    L_drug_disease = adj_to_list_noidentical('~/Desktop/final_project/DTINet/data/mat_drug_disease.txt',' ',2220,0,3)   
    L_protein_disease = adj_to_list_noidentical('~/Desktop/final_project/DTINet/data/mat_protein_disease.txt',' ',2220,707,4)    
    L_drug_se = adj_to_list_noidentical('~/Desktop/final_project/DTINet/data/mat_drug_se.txt',' ',7823,0,5)    
    L_drug_protein = adj_to_list_noidentical('~/Desktop/final_project/DTINet/data/mat_drug_protein.txt',' ',707,0,6)
        
    
    #create dataframes for lists and concatenate them
           
    D_drug_drug = create_dataframe(L_drug_drug) 
    print_length(D_drug_drug,"drug_drug length is")
    
    D_protein_protein = create_dataframe(L_protein_protein)
    print_length(D_protein_protein,"protein-protein length is")
    
    D_drug_disease = create_dataframe(L_drug_disease)
    print_length(D_drug_disease,"drug-disease length is")  
        
    D_protein_disease = create_dataframe(L_protein_disease)
    print_length(D_protein_disease,"protein_disease length is")
    
    D_drug_se = create_dataframe(L_drug_se)
    print_length(D_drug_se,"drug_se length is")  
    
    D_drug_protein = create_dataframe(L_drug_protein)
    print_length(D_drug_protein,"drug_protein length is")
    
    #create dataset and save to .csv file
    
    frames = [D_drug_drug, D_protein_protein, D_drug_disease, D_protein_disease,D_drug_se,D_drug_protein]
    
    main_dataset = pd.concat(frames)
    main_dataset.to_csv('~/Desktop/DTINet/data/generated_data/main.csv', sep=' ',index=False, header=None)
    
    #read main.csv and add Edge ID column to it and fill with index number then save to main_db.csv
    
    main = pd.read_csv('~/Desktop/DTINet/data/generated_data/main.csv',sep=' ')
    indexN = main.index.values
    main['Edge ID'] = indexN
    main.to_csv('~/Desktop/DTINet/data/generated_data/main_db.csv', sep=' ',index=False, header=None)


    
if __name__ == "__main__":
    main()












