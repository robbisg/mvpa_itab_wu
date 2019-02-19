import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

from pyitab.utils.matrix import copy_matrix, array_to_matrix
from pyitab.analysis.results import get_results, filter_dataframe, \
        get_permutation_values, df_fx_over_keys
from pyitab.plot.connectivity import plot_connectivity_circle_edited

##### Get full results from dir `id_`
path =  "/media/robbis/DATA/fmri/working_memory/0_results"

id_ = "wm_permutation_final"
dataframe = get_results(path, 
                        dir_id=id_, 
                        field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'])


# Average across cross-validation
keys = ['band', 'targets', 'permutation', "k"]
table = pd.pivot_table(dataframe, 
                       values='accuracy', 
                       index=keys, 
                       aggfunc=np.mean).reset_index()

keys.remove("permutation")
options = {k : np.unique(table[k]) for k in keys}


keys, values = options.keys(), options.values()
opts = [dict(zip(keys,[it for it in items])) for items in product(*values)]


# Permutation dataframe
p_values = []
for item in opts:
        
    cond_dict = {k: v for k, v in item.items()}
    item = {k: [v] for k, v in item.items()}
    
    df_ = dataframe.copy()
    data_ = table.copy()
    
    
    data_ = filter_dataframe(data_, item)
    item.update({'permutation':[0]})
    df_ = filter_dataframe(df_, item)
      
    
    df_avg = np.mean(df_['accuracy'].values)
    
    p = (np.count_nonzero(data_['accuracy'].values > df_avg) + 1) / 100.
    
    cond_dict['accuracy_perm'] = np.mean(data_['accuracy'].values)
    cond_dict['accuracy_true'] = np.mean(df_['accuracy'].values)               
    cond_dict['p_value'] = p
    
    p_values.append(cond_dict)
    
p_df = pd.DataFrame(p_values)

#######################
# Plot of accuracies #

id = "wm_mpsi_rest_norm"
dataframe = get_results(path, 
                        dir_id=id, 
                        field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'])

keys = ['band', 'targets', "k"]
table = pd.pivot_table(dataframe, 
                       values='accuracy', 
                       index=keys, 
                       aggfunc=np.mean).reset_index()

# This is the plot line
g = sns.catplot(x="k", y="accuracy", hue="targets", col="band", data=dataframe,
                   kind="point", palette="Set2", size=6, aspect=.75, vmin=.45, vmax=1.)

g.axes[0][0].set_ylim(.45, .7)


# Average across-folds
df_avg = pd.pivot_table(dataframe, 
                        values='accuracy', 
                        index=['band', 'targets', 'permutation', 'k', 'prepro'], 
                        aggfunc=np.mean).reset_table()
                        
                        
df_abs = filter_dataframe(dataframe, 
                          {'permutation':[0], 'prepro':['abs_detrending_feature_norm_sample_slicer']})

                
                
######################################
# Plot imports
from matplotlib.colors import ListedColormap
from mne.viz import circular_layout

# Load data
dataframe = get_results(path, 
                        dir_id="wm_not_absolute", 
                        field_list=['estimator__fsel', 'prepro', 'sample_slicer'],
                        result_keys=['features', 'weights'], 
                        )

# Get only non-permutation results
df_real = filter_dataframe(dataframe, {'permutation':[0]})
# Sum the selection of different features
df_feat = df_real.groupby(['band','targets','k'])['features'].apply(lambda x: np.vstack(x).sum(0))
# Average across different k-values: 
df_k_feat = df_feat.reset_index().groupby(['band','targets'])['features'].apply(lambda x: np.vstack(x).mean(0))

# This is a weighted average
ks = np.unique(df_feat.reset_index()['k'].values)[::-1]
fx = lambda x: np.average(np.vstack(x), 0, weights=ks)
df_k_feat = df_feat.reset_index().groupby(['band','targets'])['features'].apply(fx)
df = df_k_feat.reset_index()

# Connectome labels loading
labels = np.loadtxt("/home/robbis/ROI_MNI_V4.txt", dtype=np.str_)
node_names = labels.T[1][:99]
node_idx = np.argsort(np.array([node[-1] for node in node_names]))

node_angles = circular_layout(node_names.tolist(), 
                                node_names[node_idx].tolist(), 
                                start_pos=90, 
                                group_boundaries=[0, len(node_names) / 2.+1])


# Plot of the connectivity circle of selected features
for band in np.unique(df['band'].values):
    for condition in np.unique(df['targets'].values):
        features = filter_dataframe(df, selection_dict={'band':[band],
                                                        'targets':[condition]}).reset_index()
        matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)

        f, a = plot_connectivity_circle_edited(matrix, 
                                                node_names=node_names, 
                                                con_thresh=70, 
                                                node_angles=node_angles,
                                                colormap='magma',
                                                node_colors=sns.husl_palette(2),
                                                title="%s %s" % (band, condition)
                                                )
        #f.savefig("/home/robbis/wm_absolute_%s_%s.png" % (band, condition), facecolor='black')


# Generation of feature selection distribution from permutation
df_permutation = filter_dataframe(df, selection_dict={'permutation':['!0']})
df_perm_features = pd.DataFrame(df_permutaton.features.values.tolist(), 
                                columns=['feature_%d' %(i) for i in range(df_permutaton.features.values[0].shape[0])])


keys_permutation = pd.DataFrame([row[:-1] for row in df_permutation.values.tolist()], 
                                columns=df.permutation.keys()[:-1])

df_permutation_feat = pd.concat(df_perm_features, keys_permutation)


arr = []
for b, band in enumerate(np.unique(df['band'].values)):
    for c, condition in enumerate(np.unique(df['targets'].values)):

        selection_dict = {'band':[band], 'targets':[condition]}

        permutation = filter_dataframe(df_permutation_feat, selection_dict=selection_dict).reset_index()
        features = filter_dataframe(df, selection_dict=selection_dict).reset_index()

        permutation_ = np.float_(permutation.values[...,1:-3])
        features_ = features['features'].values[0]

        i = c + b/0.5 * len(np.unique(df['targets'].values))

        threshold = permutation_.mean(0) + 3 * permutation_.std(0)
        features_thres = features_ > threshold
        arr.append(features_thres)
        pos = np.nonzero(features_thres)
        pl.scatter(np.ones_like(pos)*i, pos[0], s=100*(features_[features_thres] / 70)**5.)

features_arr = np.array(arr).sum(0)





for b, band in enumerate(np.unique(df['band'].values)):
    for c, condition in enumerate(np.unique(df['targets'].values)):
        pl.figure()

        features = filter_dataframe(df, selection_dict={'band':[band],
                                                        'targets':[condition]}).reset_index()
                



# Plot of brain
mask = ni.load(os.path.join(path, '1_single_ROIs', 'ROI_MNI_v4.nii'))
labels = np.loadtxt(os.path.join(path, '1_single_ROIs', 'ROI_MNI_v4.txt'), dtype=np.str_)
mask_data = mask.get_data()

# Compute mni coordinates
mni_centers = []
for i, value in enumerate(np.unique(mask_data)[1:]):
    roi_mask = mask_data == value
    center = np.mean(np.nonzero(roi_mask), axis=1)
    mni_center = center * np.diag(mask.affine)[:-1] + mask.affine[:-1,-1]
    mni_centers.append(mni_center)

# Plot of brains!
from nilearn.plotting import plot_connectome
for band in np.unique(df['band'].values):
    for condition in np.unique(df['targets'].values):
        features = filter_dataframe(df, selection_dict={'band':[band],
                                                        'targets':[condition]}).reset_index()
        matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)

        node_size = matrix * np.int_(matrix > 70)

        plot_connectome(matrix, 
                        mni_centers[:99], 
                        node_color=sns.husl_palette(2), 
                        node_size=node_size.sum(0)*2, 
                        edge_cmap='magma',
                        edge_vmin=70.,
                        edge_vmax=np.max(matrix),
                        edge_threshold=70.,
                        figure=pl.figure(figsize=(13,9)),
                        title="%s %s" % (band, condition), 
                        black_bg=True,
                        output_file="/home/robbis/wm_not_absolute_brain_%s_%s.png" % (band, condition),
                        node_kwargs={'alpha':0.95},
                        edge_kwargs={'alpha':0.8})



# Plot of both brains and circles
mni_centers_power = mni_centers[:99][mask_labels]
for band in np.unique(df['band'].values):
    for condition in np.unique(df['targets'].values):
        features = filter_dataframe(df, selection_dict={'band':[band],
                                                        'targets':[condition]}).reset_index()
        node_size = features['features'].values[0]
        node_size = node_size * np.int_(node_size > 70)

        matrix = np.ones((len(node_size), len(node_size)))

        plot_connectome(matrix, 
                        mni_centers_power, 
                        node_color=sns.husl_palette(2), 
                        node_size=node_size*4, 
                        edge_cmap='cubehelix',
                        edge_vmin=70.,
                        edge_vmax=75,
                        edge_threshold=70.,
                        figure=pl.figure(figsize=(13,9)),
                        title="%s %s" % (band, condition), 
                        black_bg=True,
                        #output_file="/home/robbis/wm_power_brain_%s_%s.png" % (band, condition),
                        node_kwargs={'alpha':0.75},
                        edge_kwargs={'alpha':0.8})

        matrix = np.random.rand(88,88) * 1
        f, a = plot_connectivity_circle_edited( matrix,
                                                node_size=node_size**9/np.max(node_size)**8,
                                                node_names=node_names[mask_labels], 
                                                node_thresh=70., 
                                                node_angles=node_angles[mask_labels],
                                                colormap='cubehelix',
                                                node_colors=sns.husl_palette(2),
                                                title="%s %s" % (band, condition),
                                                vmin=10,
                                                vmax=50,
                                                )

        #f.savefig("/home/robbis/wm_power_%s_%s.png" % (band, condition), facecolor='black')


#########################################
# Plots 17102018
from pyitab.utils.atlas import get_atlas_info
info = get_atlas_info('aal_meg')
labels, colors, node_idx, coords, networks, node_angles = info
from pyitab.plot.connectivity import plot_connectivity_lines
# MPSI
# Node frequency
dataframe = get_results(path, 
                        dir_id="wm_mpsi_detr_perm", 
                        field_list=['estimator__fsel', 'prepro', 'sample_slicer'],
                        result_keys=['features', 'weights'], 
                        filter={'permutation':[0]}
                        )

df_features = df_fx_over_keys(dataframe, 
                              keys=['targets', 'k', 'band'], 
                              fx=lambda x: np.vstack(x).sum(0))

df_features_avg = df_fx_over_keys(df_features, 
                              keys=['targets', 'band'], 
                              fx=lambda x: np.vstack(x).mean(0))



names = labels.T[1][:99]
color_array = np.array([colors[i%2] for i, _ in enumerate(names)])
array_list = []
titles = []
for band in np.unique(dataframe['band'].values):
    for condition in np.unique(dataframe['targets'].values):
            
        selection_dict = {'band':[band], 'targets':[condition]}
        features = filter_dataframe(df_features_avg, selection_dict=selection_dict)
        matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)
        matrix /= 75.
        node_size = matrix.sum(0) / 89.

        array_list.append(node_size.squeeze())
        titles.append(band)

        plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                names[node_idx],
                                node_colors=color_array[node_idx],
                                node_position=node_angles[node_idx],
                                con_thresh=0.5, 
                                kind='circle', 
                                facecolor='white', 
                                colormap='magma_r',
                                title=band)


barplot_nodes(array_list, names, color_array, titles)





######## 23 October 2018 ######
from pyitab.plot.nodes import barplot_nodes


mask_data = loadmat("/media/robbis/DATA/fmri/working_memory/sub_01/meg/connectivity_matrix.mat")
mask_ = np.sum(mask_data['data'], axis=0)                             
mask_node = mask_.sum(0)
mask_node = mask_node != 0

from pyitab.utils.atlas import get_atlas_info
info = get_atlas_info('aal_meg')
labels, colors, node_idx, coords, networks, node_angles = info
names = labels.T[1][:99]
color_array = np.array([colors[i%2] for i, _ in enumerate(names)])

dirs = ["wm_mpsi_norm_abs", "wm_mpsi_norm_sign", "wm_mpsi_norm_detr"]
dirs = ["wm_power_full_k", "wm_power_norm"]

for dir_id in dirs:

        dataframe = get_results(path, 
                                dir_id=dir_id, 
                                field_list=['estimator__fsel', 'estimator__clf', 'cv', 'sample_slicer'])
        
        f = sns.relplot(x="k", y="accuracy", col="band", hue="targets",
             height=5, aspect=.75, facet_kws=dict(sharex=False),
             kind="line", legend="full", data=dataframe)

        f.axes[0][0].set_ylim(.4, .75)

        f.savefig("/media/robbis/DATA/fmri/working_memory/figures/%s.png" % (dir_id))


################################## POWER ########################################
titles = {
        "wm_power_full_k": "Power Plain",
        "wm_power_norm": "Power #Parcel Normalization",

                }

for dir_id in dirs:
        dataframe = get_results(path, 
                                dir_id=dir_id, 
                                field_list=['estimator__fsel', 'prepro', 'sample_slicer'],
                                result_keys=['features', 'weights'], 
                                filter={'permutation':[0]}
                                )

        df_features = df_fx_over_keys(dataframe, 
                                        keys=['targets', 'k', 'band'], 
                                        fx=lambda x: np.vstack(x).sum(0))

        df_features_avg = df_fx_over_keys(df_features, 
                                        keys=['targets', 'band'], 
                                        fx=lambda x: np.vstack(x).mean(0))


        data = dict()

        # We build data getting features and projecting on nodes
        for condition in np.unique(dataframe['targets'].values):
                for band in np.unique(dataframe['band'].values):
                        selection_dict = {'band':[band], 'targets':[condition]}
                        features = filter_dataframe(df_features_avg, selection_dict=selection_dict)
                        #matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)

                        node_size = features['features'].values[0]
                        node_size_full = np.zeros(99, dtype=np.float)
                        node_size_full[mask_node] = node_size

                        node_size_full /= 75.
                        
                        key = "band: %s | condition: %s"%(band, condition)
                        data[key] = node_size_full.squeeze()


        b = barplot_nodes(list(data.values()), 
                          names, 
                          color_array, 
                          list(data.keys()), 
                          n_cols=4, 
                          n_rows=3, 
                          text_size=10,
                          title=titles[dir_id],
                          xmin=0.7)

        b.savefig("/media/robbis/DATA/fmri/working_memory/figures/nodes_%s.png" % (dir_id))


################################# MPSI #############################################à

dirs = ["wm_mpsi_norm_abs", "wm_mpsi_norm_sign", "wm_mpsi_norm_detr"]
titles = ["MPSI Absolute", "MPSI Sign", "MPSI Plain"]

titles = dict(zip(dirs, titles))


for dir_id in dirs:
        dataframe = get_results(path, 
                                dir_id=dir_id, 
                                field_list=['estimator__fsel', 'prepro', 'sample_slicer'],
                                result_keys=['features', 'weights'], 
                                filter={'permutation':[0]}
                                )

        df_features = df_fx_over_keys(dataframe, 
                                        keys=['targets', 'k', 'band'], 
                                        fx=lambda x: np.vstack(x).sum(0))

        df_features_avg = df_fx_over_keys(df_features, 
                                        keys=['targets', 'band'], 
                                        fx=lambda x: np.vstack(x).mean(0))


        data = dict()

        
        for condition in np.unique(dataframe['targets'].values):
                for band in np.unique(dataframe['band'].values):       
                        selection_dict = {'band':[band], 'targets':[condition]}
                        features = filter_dataframe(df_features_avg, selection_dict=selection_dict)
                        matrix = array_to_matrix(features['features'].values[0], copy=True, diagonal_filler=0.)
                        matrix /= 75.
                        node_size = matrix.sum(0) / 89.
                        
                        key = "band: %s | condition: %s"%(band, condition)
                        data[key] = node_size.squeeze()

                        f = plot_connectivity_lines(matrix[node_idx][:,node_idx], 
                                                    names[node_idx],
                                                    node_colors=color_array[node_idx],
                                                    node_position=node_angles[node_idx],
                                                    con_thresh=0.85, 
                                                    kind='circle', 
                                                    facecolor='white', 
                                                    colormap='magma_r',
                                                    title="%s | %s" % (titles[dir_id], key))

                        title_fig = "connection_%s_%s_%s.png" %(dir_id, band, condition)
                        f.savefig("/media/robbis/DATA/fmri/working_memory/figures/%s" % (title_fig))


        b = barplot_nodes(list(data.values()), 
                          names, 
                          color_array, 
                          list(data.keys()), 
                          n_cols=4, 
                          n_rows=3, 
                          text_size=10,
                          title=titles[dir_id],
                          xmin=0.0)
        b.savefig("/media/robbis/DATA/fmri/working_memory/figures/nodes_%s.png" % (dir_id))

################ Plots ##################
from pyitab.utils.atlas import get_atlas_info
from sklearn.preprocessing import minmax_scale
info = get_atlas_info('aal_meg')
labels, colors, node_idx, coords, networks, node_angles = info
names = labels.T[1][:99]
mni_coord = coords[:99]


dirs = [
        #"wm_mpsi_norm_abs", 
        "wm_mpsi_norm_sign", 
        #"wm_mpsi_norm_detr",
        #"wm_power_full_k",
        "wm_power_norm"
        ]

def mpsi_lambda(x):
        array = np.vstack(x).mean(0) / 75.
        matrix = array_to_matrix(array, 
                                 copy=True, 
                                 diagonal_filler=0.)
        nodes = matrix.sum(0) / 89.
        return nodes

def power_lambda(x):
        mask_data = loadmat("/media/robbis/DATA/fmri/working_memory/sub_01/meg/connectivity_matrix.mat")
        mask_ = np.sum(mask_data['data'], axis=0)                             
        mask_node = mask_.sum(0)
        mask_node = mask_node != 0
        node_size = np.vstack(x).mean(0) / 75.
        node_size_full = np.zeros(99, dtype=np.float) 
        node_size_full[mask_node] = node_size

        return node_size_full



dataframe_full = dict()
for dir_id in dirs:
        dataframe = get_results(path, 
                        dir_id=dir_id, 
                        field_list=['estimator__fsel', 'prepro', 'sample_slicer'],
                        result_keys=['features'], 
                        filter={'permutation':[0]}
                        )

        df_features = df_fx_over_keys(dataframe, 
                                        keys=['targets', 'k', 'band'], 
                                        fx=lambda x: np.vstack(x).sum(0))
        
        if "mpsi" in dir_id:


                df_features_avg = df_fx_over_keys(df_features, 
                                                keys=['targets', 'band'], 
                                                fx=mpsi_lambda)
        else:
                df_features_avg = df_fx_over_keys(df_features, 
                                                keys=['targets', 'band'], 
                                                fx=power_lambda)

        dataframe_full[dir_id] = df_features_avg

colors = {0: 'k', 1:'yellow', 2:'snow', -1:'red'}
#colors = {0: 'white', 1:'gold', 2:'gray', -1:'salmon'}

bands = ['alpha', 'beta', 'theta', 'gamma']
conditions = ['0back_2back', '0back_rest', 'rest_2back']

fig = pl.figure(figsize=(15,12), facecolor='black')
i = 0
for band in bands:
    for condition in conditions:
        dfs = []
        sizes = []
        for key in ["wm_mpsi_norm_sign", "wm_power_norm"]:
            df = dataframe_full[key]
            df = filter_dataframe(df, targets=[condition], band=[band])

            values = df['features'].values[0]
            features = np.int_(values > (values.mean()+1.15*values.std()))

            dfs.append(features)
            scaled_values = minmax_scale(values)
            sizes.append(scaled_values)

        features_boolean = (dfs[1] + dfs[0])*((dfs[1] - dfs[0])+(dfs[1]*dfs[0]))
        node_size = (dfs[1]*sizes[1] + dfs[0]*sizes[0])*50

        color_array = [colors[f] for f in features_boolean]

        print(names[node_size > 0])

        ax = fig.add_subplot(4, 3, i+1)
        i += 1
        ax.set_title("%s | %s" % (band, condition), color='white')
        plot_connectome(np.zeros((99, 99)), 
                        mni_coord, 
                        node_color=color_array, 
                        node_size=node_size, 
                        edge_cmap='Pastel2',
                        edge_vmin=0.,
                        edge_vmax=1,
                        edge_threshold=70.,
                        #figure=pl.figure(figsize=(13,9)),
                        axes=ax,
                        #title="%s %s" % (band, condition), 
                        black_bg=True,
                        #output_file="/home/robbis/wm_power_brain_%s_%s.png" % (band, condition),
                        node_kwargs={'alpha':0.75},
                        edge_kwargs={'alpha':0.8})



for key, df in dataframe_full.items():
	for values in df['features'].values:
		nodes = np.count_nonzero(values > (values.mean()+values.std()))
		print(nodes)


dataframe_full = dict()


def probability(x):
        return np.int_(np.mean(x) > .55)


frames = []
for dir_id in dirs:

        dataframe = get_results(path, 
                                dir_id=dir_id, 
                                field_list=['estimator__fsel', 'cv', 'sample_slicer'])
        
        frames.append(dataframe)

df_merged = pd.concat(frames)


task_dict = {'71cff912-51fb-4b15-860c-127600c42cf3':'Power', 
             '940cee07-396b-4bd1-84fb-07aa7e714eaf':'MPSI'}

task = [task_dict[id_] for id_ in df_merged['id'].values]
df_merged['measure'] = task

f = sns.relplot(x="k", y="score_accuracy", row="band", col="measure", hue="targets",
        height=5, aspect=.75, facet_kws=dict(sharex=False),
        kind="line", legend="full", data=df_merged)

f.axes[0][0].set_ylim(.4, .85)
#




for i, t in enumerate(['MPSI', 'Power']):
    for j, band in enumerate(np.unique(df_merged['band'].values)):
        for k, target in enumerate(np.unique(df_merged['targets'])):
            
            ax = f.axes[j][i]
            
            df = filter_dataframe(df_merged, measure=[t], band=[band], targets=[target])
            df_avg = df_fx_over_keys(df, 
                                     attr='score_accuracy',
                                     keys=['k'], 
                                     fx= np.mean)
            values = np.int_(df_avg['score_accuracy'].values >= .55)
            kk = df_avg['k'].values
            values = values * (.80 + k/50.)
            values[values == 0] = np.nan
            ax.plot(kk, values, 'o', c=ax.get_children()[k+3].get_color())

f.savefig("/media/robbis/DATA/fmri/working_memory/figures/ohbm.png", dpi=150)


