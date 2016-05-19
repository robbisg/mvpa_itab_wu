import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as pl

ds_ = np.loadtxt('/media/robbis/DATA/fmri/loris/data.csv', delimiter=',', dtype=np.str_)

column_legend = {'Paziente':0,
                 'colonna_controlli':1,
                'patologia':2,
                'sede ant':3,
                'sede post':4,
                'numero voxel':5,
                'volume lesione':6,
                'distanza in sigma':7,
                'broca x':8,
                'broca y':9,
                'broca z':10,
                'beta':11,
                'broca_TALA':12,
                'distanza_radq_broca_TALA':13,
                'broca_contr':14,
                'RADQ_broca_contr':15,
                'tpj x':16,
                'tpj y':17,
                'tpj z':18,
                'beta':19,
                'tpj_tala':20,
                'radq_tpj_tala':21,
                'd_tpj_controlli':22,
                'radq_tpj_con':23,
                'sts x':24,
                'sts y':25,
                'sts z':26,
                'beta':27,
                'sts_tala':28,
                'radq_sts':29,
                'sts_controlli':30,
                'radq_sts_cont':31,                 
                 }




index_list = ['broca x','broca y','broca z','tpj x','tpj y','tpj z','sts x','sts y','sts z']
x_indexes = [column_legend[i] for i in index_list]


X = ds_[1:, x_indexes]
y = ds_[1:, column_legend['patologia']]
y1 = ds_[1:, column_legend['colonna_controlli']]

data_frame = dict(zip(index_list, np.int_(X.T)))
data_frame['zpat'] = np.int_(y)
data_frame['zcon'] = np.int_(y1)

df = pd.DataFrame(data_frame)

g = sns.PairGrid(df, vars=index_list)



def pair_plot(df, label, index_labels):
    
    g = sns.PairGrid(df, vars=index_labels, hue=label)
    g = g.map_diag(sns.kdeplot, lw=3, legend=False)
    g = g.map_upper(pl.scatter)
    g = g.map_lower(pl.scatter)
    g = g.add_legend()





