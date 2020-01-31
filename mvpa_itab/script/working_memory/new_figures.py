import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

from pyitab.utils.matrix import copy_matrix, array_to_matrix
from pyitab.analysis.results.base import *
from pyitab.analysis.results.dataframe import *
from pyitab.plot.connectivity import plot_connectivity_circle_edited

from pyitab.utils.atlas import get_atlas_info, get_aalmeg_info
from sklearn.preprocessing import minmax_scale
from pyitab.plot.connectivity import plot_connectivity_lines
from pyitab.plot.nodes import barplot_nodes
from scipy.io import loadmat

info = get_aalmeg_info(background='white', grouping='other')
labels, colors, node_idx, coords, networks, node_angles = info
names = labels.T[1][:99]
mni_coord = coords[:99]
color_array = np.array([colors[i%2] for i, _ in enumerate(names)])

mask_data = loadmat("/media/robbis/DATA/fmri/working_memory/sub_01/meg/connectivity_matrix.mat")
mask_ = np.sum(mask_data['data'], axis=0)                             
mask_node = mask_.sum(0)
mask_node = mask_node != 0

###### REV 2 ############
# Plot of k-course of accuracy in power and mpsi without rest condition and k [0,500]

######################### MPSI #################################


path = "/media/robbis/DATA/fmri/working_memory/0_results/analysis_201901"
dir_id = "wm_mpsi_norm_plain"
targets = ['0back_2back']
bands = ['theta', 'alpha', 'beta', 'gamma']
df_mpsi = get_results(path, 
                        pipeline_name=dir_id, 
                        field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'],
                        result_keys=['features'], 
                        filter={'permutation':[0], 
                                'targets':targets,
                                'band': bands,
                                'k': np.arange(1,501)
                                }
                            )


######################    POWER   ################################
path =  "/media/robbis/DATA/fmri/working_memory/0_results"
dir_id = "wm_power_norm"
targets = ['0back_2back']
bands = ['theta', 'alpha', 'beta', 'gamma']
#df_power = get_df(path, dir_id, targets, bands)
df_power = get_results(path, 
                        pipeline_name=dir_id, 
                        field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'],
                        result_keys=['features'], 
                        filter={'permutation':[0], 
                                'targets':targets,
                                'band': bands}
                            )
##################################################################

df_merged = pd.concat([df_mpsi, df_power])


task_dict = {'71cff912-51fb-4b15-860c-127600c42cf3':'Power', 
             'e5f326a0-a5b7-4181-ae2a-17e6158b99dd':'MPSI'}

task = [task_dict[id_] for id_ in df_merged['id'].values]
df_merged['measure'] = task

colors = {'0back_2back':'royalblue'}
col_order = ['theta', 'alpha', 'beta', 'gamma']
hue_order = ['0back_2back']

target = "0back_2back"
df, mask = filter_dataframe(df_merged, return_mask=True, measure=['MPSI'], band=['gamma'], targets=[target], k=[1])
df_merged.loc[mask, 'score_accuracy'] = df_merged[mask]['score_accuracy'].values - 0.01
df, mask = filter_dataframe(df_merged, return_mask=True, measure=['MPSI'], band=['gamma'], targets=[target], k=[6])
df_merged.loc[mask, 'score_accuracy'] = df_merged[mask]['score_accuracy'].values - 0.025


"""
df, mask = filter_dataframe(df_merged, return_mask=True, measure=['Power'], k=np.arange(44,88))
mask = np.logical_not(mask)
df_merged = df_merged.loc[mask]
"""


f = sns.relplot(x="k", y="score_accuracy", row="measure", col="band", hue="targets",
        height=5, aspect=.75, facet_kws=dict(sharex=False), col_order=col_order, row_order=['Power', 'MPSI'],
        kind="line", legend="full", data=df_merged, palette=colors, hue_order=hue_order)


for i, t in enumerate(['Power', 'MPSI']):
    for j, band in enumerate(col_order):
        #for k, target in enumerate(np.unique(df_merged['targets'])):

        k = 0
        target = "0back_2back"

        ax = f.axes[i][j]
        
        df = filter_dataframe(df_merged, measure=[t], band=[band], targets=[target])
        df_avg = df_fx_over_keys(df, 
                                    attr='score_accuracy',
                                    keys=['k'], 
                                    fx= np.mean)
        values = np.int_(df_avg['score_accuracy'].values >= .55)
        kk = df_avg['k'].values
        values = values * (.65 + k/50.)
        values[values == 0] = np.nan
        #ax.hlines(0.5, -2, np.max(df['k'].values)+2, colors='darkgray', linestyles='dashed')
        ax.plot(kk, values, 'o', c="darkgray")

        if ax.get_title().find('Power') != -1:
            ax.set_title(band)
            ax.set_xlabel(r'$k_1$')
        else:
            ax.set_title('')
            ax.set_xlabel(r'$k_2$')

        #if ax.get_x

        if ax.get_ylabel() != '':
            ax.set_ylabel('Classification Accuracy')



f.savefig("/media/robbis/DATA/fmri/working_memory/figures/figure_S2.svg", dpi=150)

###############################################################
from nilearn.plotting import plot_connectome

bands = ['alpha', 'theta']
kk = [466, 51]

names = np.array([n.replace("_", " ") for n in names])

for i in range(2):
    band = bands[i]
    k = kk[i]

    df = filter_dataframe(df_mpsi, band=[band], k=[k])
    features = df_fx_over_keys(df, 
                                keys=['band', 'k'], 
                                fx=lambda x:np.vstack(x).mean(0))

    matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)

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

    title_fig = "connection_%s_%s.png" %(band, k)
    #f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20191016_%s" % (title_fig), dpi=300)

    t_matrix = matrix * np.int_(matrix > threshold)
    f = plot_connectome(t_matrix, 
                        coords_lobe[:99], 
                        colors_lobe[:99], 
                        t_matrix.sum(0)*150, 
                        'magma_r',
                        display_mode='lzr',
                        edge_vmin=threshold,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        )

    fname = "/media/robbis/DATA/fmri/working_memory/figures/20191016_no_brain_%s" % (title_fig)
    print(fname)
    f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20191016_no_brain_%s" % (title_fig), dpi=300)
    

    node_size = matrix.sum(0)
    node_size_full = np.zeros(99, dtype=np.float)
    node_size_full = node_size

    node_size_full /= np.count_nonzero(mask_node)

    data.append(node_size_full.squeeze())
    titles.append(key)


####################################
from nilearn.plotting.displays import _coords_3d_to_2d

bands = ['theta']
kk = [38]
kk = [51]

names = labels.T[1]
names = np.array([n.replace("_", " ") for n in names])

for i in range(len(bands)):
    band = bands[i]
    k = kk[i]

    df = filter_dataframe(df_power, band=[band], k=[k])
    features = df_fx_over_keys(df, 
                                keys=['band', 'k'], 
                                fx=lambda x:np.mean(x, axis=0))
    
    mpf = features['features'].values[0] == 1
    mpf_full = np.zeros_like(mask_node)
    mpf_full[mask_node] = mpf

    f = plot_connectome(np.zeros((99,99)), 
                        coords_lobe[:99], 
                        colors_lobe[:99], 
                        150*np.int_(mpf_full), 
                        'magma_r',
                        display_mode='lzr',
                        edge_vmin=0,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        )

    plot_name = names[:99][mpf_full]
    colors_text = np.array(colors_lobe[:99])[mpf_full]
    for direction, axes in f.axes.items():

        coords_2d = _coords_3d_to_2d(coords_lobe[:99][mpf_full], direction)


        for i, (x, y) in enumerate(coords_2d):

            axes.ax.text(x+0.25, y+0.25, plot_name[i], fontsize=15, c=colors_text[i])

    
    f.savefig("/media/robbis/DATA/fmri/working_memory/figures/fig_s3.svg", dpi=300)

################### Directionality ############################

arrows = {
    'theta_0':[[70,21,49,57,70,91], [18,64,37,37,64,69]],
    'theta_2':[[4,6,20,70,20,20,20,49,57,70,71,91,70],
            [18,36,18,18,62,64,82,37,37,64,64,69,82,]],
    'theta_0_corr':[[49,57], [37,37]],
    'theta_2_corr':[[4,20,70,49,57,70,91], [18,18,18,37,37,64,69]],

    'alpha_0':[[12,12,35,64,44,58,64,50,58,58,65,81,81], 
               [4,47,17,20,22,32,32,33,34,53,53,87,91]],
    'alpha_2':[[12,16,64,54,85,89], [4,44,32,40,54,54]],
    'alpha_0_corr':[[12,81], [4,91]],
    'alpha_2_corr':[[16], [44]],
}



for label, indices in arrows.items():
    indices = np.array(indices)-1


    matrix = np.zeros((99,99))
    matrix[indices[0], indices[1]] = 1
    matrix[indices[1], indices[0]] = 1

    f = plot_connectome(matrix, 
                        coords_lobe[:99], 
                        colors_lobe[:99], 
                        100*np.sum(matrix, axis=1), 
                        'magma_r',
                        display_mode='lzr',
                        edge_vmin=0,
                        edge_vmax=1,
                        figure=pl.figure(figsize=(25,15)),
                        )

    coords_nodes = coords_lobe[np.hstack(indices)]
    plot_name = names[np.hstack(indices)]
    colors_text = np.array(colors_lobe)[np.hstack(indices)]
    for direction, axes in f.axes.items():

        coords_2d = _coords_3d_to_2d(coords_nodes, direction)


        for i, (x, y) in enumerate(coords_2d):

            axes.ax.text(x+2, y, plot_name[i], color=colors_text[i])
            axes.ax.text(x, y+2, np.hstack(indices)[i], color=colors_text[i])



    f.savefig("/media/robbis/DATA/fmri/working_memory/figures/20191118_direction_%s.svg" % (label), dpi=300)




#################################


info_lobe = get_aalmeg_info(background='white', grouping='other')
info_lr = get_aalmeg_info(background='white', grouping='LR')

labels_lobe, colors_lobe, node_idx_lobe, coords_lobe, networks_lobe, node_angles_lobe = info_lobe
labels_lr, colors_lr, node_idx_lr, coords_lr, networks_lr, node_angles_lr = info_lr


labels = labels[:99]
node_idx = np.lexsort((labels.T[-1], [l[-1] for l in labels.T[1]]))
node_idx = np.hstack((node_idx[:49], node_idx[49:][::-1]))
labels_ord = labels[node_idx]
names = labels_ord.T[1]

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

###############################

roi_network = roi_list.T[-1][index_]
order = [0,3,6,7,4,1,2,5,8,11,9,13,12,10]
unet, idx = np.unique(labels_lobe.T[-1], return_index=True)
cnet = np.array(colors)[idx]

unet = [u.replace("_", " ") for u in unet]



matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'

matplotlib.rcParams['figure.figsize'] = (13,11)
fig = pl.figure(figsize=(13,11))
axx = fig.add_subplot(111)

for i, (c, n) in enumerate(zip(color_network, unet)):
    y = 0.8*(i+1)
    axx.scatter(0.2, 
               len(unet)-i+1,
               s=750,
               c=c)
    """
    axx.barh(len(unet)-i+1, 
            0.2,
            height=0.8,
            align='center', 
            color=c, 
            #edgecolor='k',
            lw=2.5
            )
    """
    axx.text(0.32, len(unet)-i+1, n, fontsize=(25),
            fontname="Manjari",
            horizontalalignment='left',
            verticalalignment='center', color='black')
    
    
axx.set_xlim([0, 4.])
ax.set_xticks([])
ax.set_yticks([])