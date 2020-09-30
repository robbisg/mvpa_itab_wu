import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

from pyitab.utils.matrix import copy_matrix, array_to_matrix
from pyitab.analysis.results.bids import filter_dataframe, get_results_bids
from pyitab.analysis.results.dataframe import apply_function, query_rows
from pyitab.plot.connectivity import plot_connectivity_circle_edited, plot_connectivity_lines

path = "/scratch/work/guidotr1/data/derivatives"
path = "/media/robbis/Seagate_Pt1/data/working_memory/data/derivatives/"

pipeline = "feature+stacked+600" # Features correct
pipeline = "dualband+bug" # Dualband
pipeline = "dualband+correct"  # Dualband with ordered features
pipeline = "dualband+correct+sparse"  # Dualband with sparse k and ordered
pipeline = "dualband+correct+full"
pipeline = "triband+bug+sparse"

dataframe = get_results_bids(path, pipeline=pipeline, 
                             field_list=['estimator__fsel', 'ds.a.task', 
                                         'ds.a.prepro', 'ds.a.img_pattern',
                                         'sample_slicer'])

dataframe = filter_dataframe(dataframe, **{'ds.a.task':['CONN']})

fig, axes = pl.subplots(1,1, figsize=(5,5))
ax = axes


k = 0
target = "0back+2back"


df = filter_dataframe(dataframe, targets=[target])
df = filter_dataframe(df, **{"ds.a.task":['CONN']})
df_avg = apply_function(df, attr='score_score', keys=['k'], fx= np.mean)
df_std = apply_function(df, attr='score_score', keys=['k'], fx= np.std)

avg = df_avg['score_score'].values[10::8]
std = (df_std['score_score'].values / np.sqrt(25))[10::8]
kk = df_avg['k'].values[10::8]

ax.plot(kk, avg, color='steelblue')
ax.fill_between(kk, avg+std, avg-std, color='steelblue', alpha=0.3)
ax.set_ylim(.45, .75)
ax.set_ylabel('Classification accuracy', fontsize=14)
ax.set_xlabel('k', fontsize=14)
ax.set_title('Multiband classification accuracy', fontsize=14)
ax.hlines(0.5, -2, np.max(df['k'].values)+2, colors='darkgray', linestyles='dashed')

fig.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/multiband.svg",
            dpi=200)

np.where( avg == np.max(avg))


############### Features ######################
df_features = get_results_bids(path, 
                               pipeline="feature+stacked+no+bug", 
                               field_list=['estimator__fsel', 
                                        'estimator__clf', 
                                        'cv', 
                                        'sample_slicer', 
                                        'ds.a.task'],
                                result_keys=['features'])

df_features['task'] = df_features['ds.a.task'].values

df = filter_dataframe(df_features, k=[28])
features = apply_function(df, keys=['k', 'task'], 
                              fx=lambda x:np.vstack(x).mean(0))

triu = np.triu_indices(99, 1)
n_elements = triu[0].shape[0]

full_features = features['features'].values[0]

full_features = loadmat("multiband-features.mat")['features']

nz = full_features != 0
threshold = full_features[nz].mean() + 3.*full_features[nz].std()

for i, band in enumerate(['alpha', 'beta', 'gamma', 'theta']):
    #features_ = features['features'].values[0][n_elements*i:n_elements*(i+1)]
    features_ = full_features[0][n_elements*i:n_elements*(i+1)]
    matrix = array_to_matrix(features_, copy=True, diagonal_filler=0.)
    
    print(band, 
          np.count_nonzero(matrix > threshold), 
          features_[features_ > threshold])
    if np.count_nonzero(matrix > threshold) == 0:
        continue
    f = plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                names,
                                node_size=220,
                                linewidth=1.5,
                                node_colors=colors,
                                node_position=node_angles,
                                con_thresh=threshold, 
                                kind='circle', 
                                facecolor='white', 
                                colormap='magma_r',
                                fontsize=17)

    f.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/%s" \
                % ("multiband_features_%s.svg" % (band)),
                dpi=300)


################## Atlas stuff ########################
from pyitab.utils.atlas import get_aalmeg_info
from mne.viz import circular_layout


mask_data = loadmat("/media/robbis/Seagate_Pt1/data/working_memory/data/sub_01/meg/connectivity_matrix.mat")
mask_ = np.sum(mask_data['data'], axis=0)                             
mask_node = mask_.sum(0)
mask_node = mask_node != 0


info_lr = get_aalmeg_info(background='white', grouping='LR')
labels_lr, colors_lr, node_idx_lr, coords_lr, networks_lr, node_angles_lr = info_lr

labels = labels_lr[:99]


node_idx = np.lexsort((labels.T[-1], [l[-1] for l in labels.T[1]]))
node_idx = np.hstack((node_idx[:49], node_idx[49:][::-1]))


labels_ord = labels[node_idx]

coords_lr_ord = coords_lr[node_idx]
names = labels_ord.T[1]
names = np.array([n.replace("_", " ") for n in names])

node_angles = circular_layout(names.tolist(),
                              names.tolist(),
                              start_between=False,
                              start_pos=90,
                              group_boundaries=[0, 49, len(names) / 2.+1],
                              group_sep=3.)

node_network = labels_ord.T[3]
networks, count = np.unique(node_network, return_counts=True)
color_network = sns.color_palette("Paired", len(networks)+1)
colors_ = dict(zip(networks, color_network[1:]))

colors = [colors_[n] for n in node_network]


########################################################
path = "/scratch/work/guidotr1/data/derivatives"
path = "/media/robbis/Seagate_Pt1/data/working_memory/data/derivatives/"

dataframes = []

pipeline = "feature+stacked+600" # Features correct

dataframe = get_results_bids(path, pipeline=pipeline, 
                             field_list=['estimator__fsel', 'ds.a.task', 
                                         'ds.a.prepro', 'ds.a.img_pattern',
                                         'sample_slicer'])
dataframes.append(dataframe)

pipeline = "dualband+correct+full"

dataframe = get_results_bids(path, pipeline=pipeline, 
                             field_list=['estimator__fsel', 'ds.a.task', 
                                         'ds.a.prepro', 'ds.a.img_pattern',
                                         'sample_slicer'])
dataframes.append(dataframe)
path = "/media/robbis/Seagate_Pt1/data/working_memory/derivatives/aalto/derivatives/"
pipeline = "triton+old"

dataframe = get_results_bids(path, pipeline=pipeline, 
                             field_list=['estimator__fsel', 'ds.a.task', 
                                         'ds.a.prepro', 'ds.a.img_pattern',
                                         'sample_slicer'])
dataframes.append(dataframe)

