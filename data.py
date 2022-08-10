import numpy as np
import pandas as pd
subjects = 77
n = 35
ones = 38



data = pd.read_csv (r'UCSFFSX6_05_02_22.csv')


to_drop = ['COLPROT',
            'VISCODE',
            'EXAMDATE',
            'VERSION',
            'update_stamp',
            'VISCODE2',
            'LONIUID',
            'IMAGEUID',
            'RUNDATE',
            'TEMPQC',
            'FRONTQC',
            'PARQC',
            'INSULAQC',
            'OCCQC',
            'BGQC',
            'CWMQC',
            'VENTQC',
            'HIPPOQC',
            'STATUS']


data.drop(to_drop, inplace=True, axis=1)

data.drop_duplicates(subset ="RID", keep = 'first', inplace=True)
data = data.set_index('RID')

colonnes = data.columns
col_sv = [x for x in colonnes if x.endswith('SV')]
data.drop(col_sv, inplace=True, axis=1)

data['OVERALLQC'] = data['OVERALLQC'].replace(['Partial'],'Alzheimer')
data['OVERALLQC'] = data['OVERALLQC'].replace(['Pass'],'NonAlzheimer')
index_with_nan = data.index[data.isnull().any(axis=1)]
data.drop(index_with_nan, inplace=True)
print(data)


mean, std = np.random.rand(), np.random.rand()





samples = []
for i in range(subjects):
    mean=0.8042317983338628
    std = 0.14772214713143283
    b = np.abs(np.random.normal(mean, std, (n,n))) % 1.0
    b_symm = (b + b.T)/2
    b_symm[np.diag_indices_from(b_symm)] = 0
    samples.append(b_symm)
samples=np.asarray(samples)
print(samples)

labels = np.zeros(subjects)
labels[:ones]  = 1
np.random.shuffle(labels)
labels=np.asarray(labels)

print(labels)

