import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

from pyitab.utils.matrix import copy_matrix, array_to_matrix
from pyitab.analysis.results import *
from pyitab.plot.connectivity import plot_connectivity_circle_edited

path = "/media/robbis/DATA/fmri/working_memory/0_results/analysis_201901"
dirs = ["wm_mpsi_norm_abs", "wm_mpsi_norm_sign", "wm_mpsi_norm_plain"]
titles = ["MPSI Absolute", "MPSI Sign", "MPSI Plain"]

titles = dict(zip(dirs, titles))


for dir_id in dirs:
        dataframe = get_results(path, 
                                dir_id=dir_id, 
                                field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'])
        
        f = sns.relplot(x="k", y="score_accuracy", col="band", hue="targets",
             height=5, aspect=.75, facet_kws=dict(sharex=False),
             kind="line", legend="full", data=dataframe)

        f.axes[0][0].set_ylim(.45, .8)
        #f.savefig("/media/robbis/DATA/fmri/working_memory/figures/2019/%s.png" % (dir_id))

#########################################################################
from pyitab.utils.atlas import get_atlas_info
from sklearn.preprocessing import minmax_scale
from pyitab.plot.connectivity import plot_connectivity_lines
from pyitab.plot.nodes import barplot_nodes
from scipy.io import loadmat

info = get_atlas_info('aal_meg')
labels, colors, node_idx, coords, networks, node_angles = info
names = labels.T[1][:99]
mni_coord = coords[:99]
color_array = np.array([colors[i%2] for i, _ in enumerate(names)])

mask_data = loadmat("/media/robbis/DATA/fmri/working_memory/sub_01/meg/connectivity_matrix.mat")
mask_ = np.sum(mask_data['data'], axis=0)                             
mask_node = mask_.sum(0)
mask_node = mask_node != 0
    
##########################################################################
# Select the results with the best accuracy for each band in 0back-2back #

def get_df(path, dir_id, targets=['0back_2back'], bands=['alpha', 'theta']):          
            
    dataframe = get_results(path, 
                            dir_id=dir_id, 
                            field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'],
                            result_keys=['features'], 
                            filter={'permutation':[0], 
                                    'targets':targets,
                                    'band': bands}
                            )


    dataframe_accuracy = df_fx_over_keys(dataframe, 
                                        keys=['targets', 'band', 'k'],
                                        attr='score_accuracy',
                                        fx=lambda x: np.mean(x))

    max_k = query_rows(dataframe_accuracy, 
                    keys=['targets', 'band'], 
                    attr='score_accuracy')

    keys = ['targets', 'band', 'k']

    selection = max_k[keys].to_dict('records')


    selection_dict = []
    for select_ in selection:
        ss = dict()
        for k, v in select_.items():
            ss[k] = [v]
        selection_dict.append(ss)


    features_df = pd.concat([filter_dataframe(dataframe, **s) for s in selection_dict])

    features_ = df_fx_over_keys(features_df, 
                                keys=['targets', 'band', 'k'], 
                                fx=lambda x:np.vstack(x).mean(0))

    dataframe_accuracy_above = dataframe_accuracy[dataframe_accuracy['score_accuracy'] >= 0.55]
    keys = ['targets', 'band', 'k']

    selection = dataframe_accuracy_above[keys].to_dict('records')

    selection_dict = []
    for select_ in selection:
        ss = dict()
        for k, v in select_.items():
            ss[k] = [v]
        selection_dict.append(ss)


    features_above = pd.concat([filter_dataframe(dataframe, **s) for s in selection_dict])

    features_above_ = df_fx_over_keys(features_above, 
                                    keys=['targets', 'band'], 
                                    fx=lambda x:np.vstack(x).mean(0))


    return pd.concat([features_, features_above_], sort=True)



######################### MPSI #################################


path = "/media/robbis/DATA/fmri/working_memory/0_results/analysis_201901"
dir_id = "wm_mpsi_norm_plain"
targets = ['0back_2back']
bands = ['theta', 'alpha']
df_mpsi = get_df(path, dir_id, targets, bands)


######################    POWER   ################################
path =  "/media/robbis/DATA/fmri/working_memory/0_results"
dir_id = "wm_power_norm"
targets = ['0back_2back']
bands = ['beta', 'alpha']

df_power = get_df(path, dir_id, targets, bands)

##################################################################

names = np.array([n.replace("_", " ") for n in names])


data = []
titles = []
for i, row in great_df.iterrows():

    band = row['band']
    condition = row['targets']
    k = row['k']

    matrix = array_to_matrix(row['features'], copy=True, diagonal_filler=0.)
    
    key = "band: %s | condition: %s | k: %d"%(band, condition, k)

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper = upper[np.nonzero(upper)]

    threshold = upper.mean() + 3.*upper.std()

    if threshold > 1:
        threshold = 0.98



    print(threshold)
    f = plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                names[node_idx],
                                node_colors=color_array[node_idx],
                                node_position=node_angles[node_idx],
                                con_thresh=threshold, 
                                kind='circle', 
                                facecolor='white', 
                                colormap='magma_r',
                                fontsize=12,
                                title=key)

    title_fig = "connection_%s_%s_%s.png" %(band, condition, k)
    f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20190212_%s" % (title_fig))
    
    node_size = matrix.sum(0)
    node_size_full = np.zeros(99, dtype=np.float)
    node_size_full = node_size

    node_size_full /= np.count_nonzero(mask_node)

    data.append(node_size_full.squeeze())
    titles.append(key)

b = barplot_nodes(data, 
                  names, 
                  color_array, 
                  titles, 
                  n_cols=2, 
                  n_rows=2,
                  selected_nodes=15,
                  text_size=12,
                  title='',
                  xmin=0.0)

b.savefig("/media/robbis/DATA/fmri/working_memory/figures/20190212_nodes_%s.png" % (dir_id))




data = []
titles = []
for i, row in df_power.iterrows():
    band = row['band']
    condition = row['targets']
    k = row['k']
    key = "band: %s | condition: %s | k: %d"%(band, condition, k)

    data.append(row['features'])
    titles.append(key.replace("_", "-"))



b = barplot_nodes(data, 
                  names, 
                  color_array, 
                  titles, 
                  n_cols=2, 
                  n_rows=2,
                  selected_nodes=15,
                  text_size=12,
                  title='',
                  xmin=0.7)

b.savefig("/media/robbis/DATA/fmri/working_memory/figures/20190212_nodes_%s.png" % (dir_id))

power = np.zeros(99)
power[mask_node] = data[0]
####################################################
from nilearn.plotting import plot_connectome



import matplotlib as mpl
import matplotlib.cm as cm

from sklearn.preprocessing import minmax_scale
color_array = np.array([colors[i%2] for i, _ in enumerate(names)])

for i, row in df_mpsi.iterrows():
    matrix = array_to_matrix(row['features'], copy=True, diagonal_filler=0.)
    node_size = matrix.sum(0) / 88.
    band = row['band']
    condition = row['targets']
    k = row['k']

    norm = mpl.colors.Normalize(vmin=node_size.min(), 
                                vmax=node_size.max())
    m = cm.ScalarMappable(norm=norm, cmap='plasma')
    colors_power = m.to_rgba(node_size)

    node_size_ = minmax_scale(node_size)

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper = upper[np.nonzero(upper)]

    threshold = upper.mean() + 3.*upper.std()

    if threshold > 1:
        threshold = 0.98



    node_size_ = node_size_ * np.bool_((matrix > threshold).sum(0))
    node_size_ = minmax_scale(node_size_)    


    plot_connectome(matrix, 
                    mni_coord, 
                    node_color=color_array, 
                    node_size=node_size_ * 250., 
                    edge_cmap='magma',
                    edge_vmin=threshold,
                    edge_vmax=np.max(matrix),
                    edge_threshold=threshold,
                    figure=pl.figure(figsize=(13,9)),
                    title="band : %s | condition : %s | k : %d" % (band, condition, k), 
                    black_bg=True,
                    output_file="/media/robbis/DATA/fmri/working_memory/figures/20190212_brain_mpsi_%s_k-%d.png" % (band, k),
                    node_kwargs={'alpha':0.9},
                    edge_kwargs={'alpha':0.8})







for i, row in df_mpsi.iterrows():
    matrix = array_to_matrix(row['features'], copy=True, diagonal_filler=0.)
    node_size = matrix.sum(0) / 88.
    band = row['band']
    condition = row['targets']
    k = row['k']

    norm = mpl.colors.Normalize(vmin=node_size.min(), 
                                vmax=node_size.max())
    m = cm.ScalarMappable(norm=norm, cmap='plasma')
    colors_power = m.to_rgba(node_size)

    node_size_ = minmax_scale(node_size)

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper = upper[np.nonzero(upper)]

    threshold = upper.mean() + 3.*upper.std()

    if threshold > 1:
        threshold = 0.98



    #node_size_ = node_size_ * np.bool_((matrix > threshold).sum(0))
    #node_size_ = minmax_scale(node_size_)    


    plot_connectome(np.zeros((99, 99)), 
                    mni_coord, 
                    node_color=colors_power, 
                    node_size=node_size_ * 250., 
                    edge_cmap='magma',
                    edge_vmin=threshold,
                    edge_vmax=np.max(matrix),
                    edge_threshold=threshold,
                    figure=pl.figure(figsize=(13,9)),
                    title="band : %s | condition : %s | k : %d" % (band, condition, k), 
                    black_bg=True,
                    output_file="/media/robbis/DATA/fmri/working_memory/figures/20190212_brain_nodes_mpsi_%s_k-%d.png" % (band, k),
                    node_kwargs={'alpha':0.9},
                    edge_kwargs={'alpha':0.8})



power_data = []
for i, row in df_power.iterrows():
    
    power = np.zeros(99)

    power[mask_node] = row['features']

    node_size = power

    power_data.append(power)

    band = row['band']
    condition = row['targets']
    k = row['k']

    norm = mpl.colors.Normalize(vmin=node_size.min(), 
                                vmax=node_size.max())
    m = cm.ScalarMappable(norm=norm, cmap='plasma')
    colors_power = m.to_rgba(node_size)

    node_size_ = minmax_scale(node_size)

    
    plot_connectome(np.zeros((99, 99)), 
                    mni_coord, 
                    node_color=colors_power, 
                    node_size=node_size_ * 250., 
                    edge_cmap='magma',
                    edge_vmin=0.,
                    edge_vmax=np.max(matrix),
                    edge_threshold=0.,
                    figure=pl.figure(figsize=(13,9)),
                    title="band : %s | condition : %s | k : %d" % (band, condition, k), 
                    black_bg=True,
                    output_file="/media/robbis/DATA/fmri/working_memory/figures/20190212_brain_nodes_power_%s_k-%d.png" % (band, k),
                    node_kwargs={'alpha':0.9},
                    edge_kwargs={'alpha':0.8})

