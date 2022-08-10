from turtle import st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate_data import simulate_data, simulate_labels
subjects = 77
n= 35
labels = simulate_labels(38, 77)
samples = []

mean, std = np.random.rand(), np.random.rand()
b = np.abs(np.random.normal(mean, std, (n,n))) % 1.0
b_symm = (b + b.T)/2
b_symm[np.diag_indices_from(b_symm)] = 0
samples.append(b_symm)

culomns=[]
lignes = []
for i in range(35):
    titre_1 = "y"+str(i)
    titre_0 = "x"+str(i)
    culomns.append(titre_1)
    lignes.append(titre_0)

print(np.shape(b_symm))
#print(culomns)
samples_df = pd.DataFrame(b_symm, columns = culomns)
labels_df = pd.DataFrame(labels, columns =['Target'])

alzheimer_disease_df = pd.concat((samples_df, labels_df), axis=1)


print(alzheimer_disease_df)
