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
path = "/media/robbis/Seagate_Pt1/data/working_memory/derivatives/aalto/derivatives/"
full_df = get_results_bids(path, pipeline="triton+old", field_list=['estimator__fsel', 
                                                                    'ds.a.task', 
                                                                    'ds.a.prepro',
                                                                    'ds.a.img_pattern',
                                                                    'sample_slicer'])
dataframe_accuracy = apply_function(full_df, 
                                    keys=['targets', 'band', 'ds.a.task', 'k'],
                                    attr='score_score',
                                    fx=lambda x: np.mean(x))
dataframe_std = apply_function(full_df, 
                                    keys=['targets', 'band', 'ds.a.task', 'k'],
                                    attr='score_score',
                                    fx=lambda x: np.std(x))

max_k = query_rows(dataframe_accuracy, 
                   keys=['targets', 'band', 'ds.a.task'], 
                   attr='score_score', fx=np.max)


#########################################################################
from pyitab.utils.atlas import get_atlas_info
from sklearn.preprocessing import minmax_scale
from pyitab.plot.connectivity import plot_connectivity_lines
from pyitab.plot.nodes import barplot_nodes
from scipy.io import loadmat

full_df = filter_dataframe(full_df, **{'ds.a.task':['CONN']})
f = sns.relplot(x="k", y="score_score", col="band", hue="targets", row='ds.a.task', 
              height=5, aspect=.75, facet_kws=dict(sharex=False), col_order=order, 
              kind="line", legend="full", data=full_df 
              )

##########################
order = ['theta', 'alpha', 'beta', 'gamma']
titles = ['Theta', 'Alpha', 'Beta', 'Gamma']


full_df = filter_dataframe(full_df, **{'ds.a.task':['CONN']})

fig, axes = pl.subplots(1, 4, sharey=True, figsize=(16,4))

for j, band in enumerate(order):
    #for k, target in enumerate(np.unique(df_merged['targets'])):
    k = 0
    target = "0back+2back"

    ax = axes[j]

    df = filter_dataframe(full_df, band=[band], targets=[target])
    df_avg = apply_function(df, attr='score_score', keys=['k'], fx= np.mean)
    df_std = apply_function(df, attr='score_score', keys=['k'], fx= np.std, ddof=1)

    avg = df_avg['score_score'].values[::5]
    std = (df_std['score_score'].values / np.sqrt(25))[::5]

    values = np.int_(df_avg['score_score'].values >= .575)[::5]
    kk = df_avg['k'].values[::5]
    values = values * (.65 + k/50.)
    values[values == 0] = np.nan
    
    ax.plot(kk, avg, c='steelblue')
    ax.fill_between(kk, avg+std, avg-std, color='steelblue', alpha=0.3)
    ax.plot(kk, values, 'o', c="darkgray")
    if j == 0:
        ax.set_ylabel('Classification accuracy')
    ax.set_xlabel('k')
    ax.hlines(0.5, -2, np.max(df['k'].values)+2, colors='darkgray', linestyles='dashed')
    ax.set_title(band)
fig.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/singleband.svg", 
            dpi=200)

##########################################################################
# Select the results with the best accuracy for each band in 0back-2back #
from scipy.io import savemat


df_features = get_results_bids(path, 
                               pipeline="triton+old", 
                               field_list=['estimator__fsel', 
                                        'estimator__clf', 
                                        'cv', 
                                        'sample_slicer', 
                                        'ds.a.task'],
                                result_keys=['features'])

df_features['task'] = df_features['ds.a.task'].values


selections = [
    {'band': ['alpha'], 'task': ['CONN'], 'k': [216]},
    {'band': ['theta'], 'task': ['CONN'], 'k': [58]},
    {'band': ['gamma'], 'task': ['CONN'], 'k': [7]},
    
    {'band': ['alpha'], 'task': ['POWER'], 'k': [72]},
    {'band': ['beta'],  'task': ['POWER'], 'k': [77]},
    {'band': ['theta'], 'task': ['POWER'], 'k': [44]},
    {'band': ['gamma'], 'task': ['POWER'], 'k': [1]},

    {'band': ['alpha'], 'task': ['POWER'], 'k': [39]},
    {'band': ['beta'],  'task': ['POWER'], 'k': [43]},
    {'band': ['theta'], 'task': ['POWER'], 'k': [34]},
    {'band': ['gamma'], 'task': ['POWER'], 'k': [1]},
]


mat_results = []

for selection in selections:
    df = filter_dataframe(df_features, **selection)
    features = apply_function(df, keys=['band', 'k', 'task'], 
                              fx=lambda x:np.vstack(x).mean(0))
    mat_results.append(features)

    # Average
    _ = selection.pop('k')
    avg_selection = selection.copy()
    df = filter_dataframe(df_features, **avg_selection)

    df_avg = apply_function(df, attr='score_score', keys=['k'], fx= np.mean)
    values = np.int_(df_avg['score_score'].values >= .55)

    indices = np.nonzero(values)[0]
    if len(indices) == 0:
        continue
    selection['k'] = indices
    df_mean = filter_dataframe(df_features, **avg_selection)
    features_ = apply_function(df_mean, keys=['band', 'task'], 
                              fx=lambda x:np.vstack(x).mean(0))

    mat_results.append(features_)


mat_results = pd.concat(mat_results)
savemat("probabilities_full.mat", {'data': mat_results.to_dict("list")})
######################### Plot of connectome #################################

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


data = []
titles = []
for i, row in mat_results[:6:2].iterrows():

    band = row['band']
    condition = row['task']
    k = row['k']

    matrix = array_to_matrix(row['features'], copy=True, diagonal_filler=0.)
    
    key = "band: %s | condition: %s | k: %d"%(band, condition, k)

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper = upper[np.nonzero(upper)]

    threshold = upper.mean() + 3.*upper.std()

    if threshold > 1:
        threshold = 0.98
    

    f = plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                names,
                                node_colors=colors,
                                node_position=node_angles,
                                con_thresh=threshold, 
                                kind='circle', 
                                facecolor='white', 
                                colormap='magma_r',
                                fontsize=12,
                                title=key)

    title_fig = "connection_%s_%s_%s.png" %(band, condition, k)
    f.savefig("/media/robbis/DATA/fmri/working_memory/figures/2020_%s" % (title_fig))


####################### Plot of brain regions #############################
from nilearn.plotting import plot_connectome
from pyitab.utils.atlas import get_aalmeg_info
from mne.viz import circular_layout
import matplotlib as mpl
import matplotlib.cm as cm

from sklearn.preprocessing import minmax_scale


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

colors_brain = [colors_[n] for n in labels.T[3][:99]]

df_mpsi = filter_dataframe(mat_results, task=['CONN'])

for i, row in df_mpsi.iterrows():


    matrix = array_to_matrix(row['features'], copy=True, diagonal_filler=0.)

    band = row['band']
    k = row['k']

    if np.isnan(k):
        k = 0
    key = "band: %s |  k: %d"%(band, k)

    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper = upper[np.nonzero(upper)]

    threshold = upper.mean() + 3.*upper.std()

    if threshold > 1:
        threshold = 0.98


    print(threshold)
    f = plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                names,
                                node_colors=colors,
                                node_position=node_angles,
                                con_thresh=threshold, 
                                kind='circle', 
                                facecolor='white', 
                                colormap='magma_r',
                                fontsize=16,
                                title=None)

    title_fig = "measure-mpsi_band-%s_k-%s_plot-circle_new.svg" %(band, k)
    #f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20191016_%s" % (title_fig), dpi=300)
    f.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/%s" % (title_fig), dpi=300)


    t_matrix = matrix * np.int_(matrix > threshold)
    f = plot_connectome(t_matrix, 
                        coords_lr[:99], 
                        colors_brain, 
                        t_matrix.sum(0)*150, 
                        'magma_r',
                        display_mode='lzr',
                        edge_vmin=threshold,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        )
    #f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20191016_%s" % (title_fig), dpi=300)
    title_fig = "measure-mpsi_band-%s_k-%s_plot-brain_new.svg" %(band, k)
    f.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/%s" % (title_fig), dpi=300)
                        


########################### Power parcels plot ##########################################
from nilearn.plotting.displays import _coords_3d_to_2d
from nilearn.plotting import plot_connectome
from pyitab.utils.atlas import get_aalmeg_info
info_lobe = get_aalmeg_info(background='white', grouping='other')

mask_data = loadmat("/media/robbis/Seagate_Pt1/data/working_memory/data/sub_01/meg/connectivity_matrix.mat")
mask_ = np.sum(mask_data['data'], axis=0)                             
mask_node = mask_.sum(0)
mask_node = mask_node != 0


labels_lobe, colors_lobe, node_idx_lobe, coords_lobe, networks_lobe, node_angles_lobe = info_lobe
names = labels_lobe.T[1][:99]
names = np.array([n.replace("_", " ") for n in names])
df_power = filter_dataframe(mat_results, task=['POWER'])

for i, row in df_power.iterrows():

    mpf = row['features'] == 1
    f = plot_connectome(np.zeros((99,99)), 
                        coords_lobe[:99], 
                        colors_lobe[:99], 
                        150*np.int_(mpf), 
                        'magma_r',
                        display_mode='lzr',
                        edge_vmin=0,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        )

    plot_name = names[:99][mpf]
    plot_number = np.nonzero(mpf)[0]
    colors_text = np.array(colors_lobe[:99])[mpf]
    for direction, axes in f.axes.items():
        coords_2d = _coords_3d_to_2d(coords_lobe[:99][mpf], direction)

        for j, (x, y) in enumerate(coords_2d):

            if direction == plot_name[j][-1].lower() or direction == 'z':
                axes.ax.text(x+2, y, plot_name[j], fontsize=15, c=colors_text[j])
                #axes.ax.text(x, y+2, str(plot_number[j]+1), fontsize=15, c=colors_text[j])
    
    title_fig = "measure-power_band-%s_k-%s_plot-parcels.svg" %(row['band'], str(row['k']))
    f.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/%s" % (title_fig), dpi=150)
############################ Directionality ########################
from nilearn.plotting import plot_connectome
from pyitab.utils.atlas import get_aalmeg_info
from mne.viz import circular_layout
import matplotlib as mpl
import matplotlib.cm as cm
from nilearn.plotting.displays import _coords_3d_to_2d
from sklearn.preprocessing import minmax_scale


info_lr = get_aalmeg_info(background='white', grouping='LR')
labels_lr, colors_lr, node_idx_lr, coords_lr, networks_lr, node_angles_lr = info_lr

labels = labels_lr[:99]

node_idx = np.lexsort((labels.T[-1], [l[-1] for l in labels.T[1]]))
node_idx = np.hstack((node_idx[:49], node_idx[49:][::-1]))

labels_ord = labels[node_idx]

coords_lr_ord = coords_lr[node_idx]
names = labels.T[1]
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

colors_brain = [colors_[n] for n in labels.T[3][:99]]


df_directions = pd.read_excel("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/significant_signs.xlsx", 
                              sheet_name=None)

for key, df in df_directions.items():

    matrix = np.zeros((99, 99))

    for i, row in df.iterrows():
        matrix[row[0]-1, row[1]-1] = 1
        matrix[row[1]-1, row[0]-1] = 0

    #matrix = copy_matrix(matrix)

    node_size = (np.int_(matrix.sum(0) != 0) + np.int_(matrix.sum(1) != 0)) * 500
    
    f = plot_connectome(matrix, 
                        coords_lr[:99], 
                        colors_brain, 
                        node_size, 
                        'magma_r',
                        display_mode='lzr',
                        #edge_vmin=threshold,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        title=key
                        )

    mpf = np.logical_or(matrix.sum(0) != 0, matrix.sum(1) != 0)

    plot_name = names[:99][mpf]
    plot_number = np.nonzero(mpf)[0]
    colors_text = np.array(colors_brain[:99])[mpf]
    for direction, axes in f.axes.items():
        coords_2d = _coords_3d_to_2d(coords_lr[:99][mpf], direction)

        for j, (x, y) in enumerate(coords_2d):

            if direction == plot_name[j][-1].lower() or direction == 'z':
                axes.ax.text(x+2, y, plot_name[j], fontsize=35, c=colors_text[j])
                #axes.ax.text(x, y+2, str(plot_number[j]+1), fontsize=15, c=colors_text[j])


    title_fig = "1-measure-mpsi_key-%s_plot-directions_nobrain.svg" %(key)
    f.savefig("/home/robbis/Dropbox/PhD/experiments/jaakko/Submission_2020/%s" % (title_fig), dpi=150)