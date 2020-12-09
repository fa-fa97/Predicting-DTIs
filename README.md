# Predicting-DTIs-using-edge2vec
In this project, we predict drug-target interactions from a heterogeneous network with various types of nodes and edges by employing a novel node embedding method called Edge2vec and by the use of an SVM classifier for predicting all of the pairs of drugs and proteins.
## Step 1
### Data gathering and preparation
The initial step for these problems is data gathering. Regarding the validation of [DTINet](https://github.com/luoyunan/DTINet) dataset has consisted of several drug-related databases, I think it would seem to have more precise results. I used these files for my project:

- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of protein names
- `se.txt`: list of side effect names
- `mat_drug_se.txt`: Drug-SideEffect association matrix
- `mat_protein_protein.txt`: Protein-Protein interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix 
- `mat_drug_drug.txt`: Drug-Drug interaction matrix
- `mat_protein_disease.txt`: Protein-Disease association matrix
- `mat_drug_disease.txt`: Drug-Disease association matrix

After this part, it turned to build our heterogeneous network. We made a CSV file containing all of the edges existing in this graph. Also, we devoted a number to each row of this file to indicate the type of the edges. Another column containing the number of each row is defined. All of these changes made our heterogeneous network. At this level, we applied these variations on the main_db.csv file.

## Step 2
### utilizing Edge2vec node embedding algorithm
In this step, we shoud employ a node embedding learning method to describe our network in a lower dimension without missing important information. In this case, due to the type of our graph which was heterogeneous, we had some choices. In some cases, we were familiar with a novel embedding method was named "Edge2vec" which the innovation of this rather than to the other algorithms was cosideration of relations' types between nodes. We applied this procedure to our network with these hyperparameters: 
