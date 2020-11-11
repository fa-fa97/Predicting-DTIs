# Predicting-DTIs-using-edge2vec
In this project, we predict drug-target interactions from a heterogeneous network with various types of nodes and edges by employing a novel node embedding method called Edge2vec and by the use of an SVM classifier for predicting all of the pairs of drugs and proteins.
## Step 1
### Data gathering and preparation
The pioneer step for these problems is data gathering. Regarding the validation of [DTINet](https://github.com/luoyunan/DTINet) dataset has consisted of several drug-related databases, I think it would seem to have more precise results. I used these files for my project:

`drug.txt`: list of drug names
`protein.txt`: list of protein names
`disease.txt`: list of protein names
`se.txt`: list of side effect names
`mat_drug_se.txt`: Drug-SideEffect association matrix
`mat_protein_protein.txt`: Protein-Protein interaction matrix
`mat_drug_protein.txt`: Drug_Protein interaction matrix 
`mat_drug_drug.txt`: Drug-Drug interaction matrix
`mat_protein_disease.txt`: Protein-Disease association matrix
`mat_drug_disease.txt`: Drug-Disease association matrix
