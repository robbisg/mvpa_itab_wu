from pyitab.analysis.results.simulations import get_results, purge_dataframe, \
    calculate_metrics, find_best_k, calculate_centroids, state_errors, \
    dynamics_errors
from pyitab.analysis.results.base import filter_dataframe
from pyitab.analysis.results.dataframe import apply_function
from pyitab.utils import make_dict_product
import pandas as pd
import numpy as np
from tqdm import tqdm


data = get_results('/media/robbis/Seagate_Pt1/data/simulations/derivatives/', 
                   pipeline='c2b+real', 
                   field_list=['sample_slicer', 
                               'n_clusters', 
                               'n_components', 
                               'ds.a.snr',
                               'ds.a.time',
                               'ds.a.states',
                               'fetch', 
                               'algorithm'],
                               
                    filter={'algorithm':['KMeans']}
                    )

df = purge_dataframe(data)

conditions = {

    'time': [0.5, 1, 1.5, 2.],
    #'num': [str(j) for j in np.arange(1, 480)],
    'snr': [10, 100, 1000, 10000],
    'algorithm': ['GaussianMixture',
                  'KMeans',
                  'AgglomerativeClustering', 
                  'SpectralClustering',
                  'MiniBatchKMeans'],
    'subject': [str(i) for i in np.arange(1, 21)]
    
}

combinations = make_dict_product(**conditions)

metrics = []
best_k = []
for options in combinations:
    df_ = filter_dataframe(df, **options)
    options = {k: v[0] for k, v in options.items()}
    df_metric = calculate_metrics(df_, fixed_variables=options)
    df_metric = df_metric.sort_values('k')
    df_k = find_best_k(df_metric)
    metrics.append(df_metric)
    best_k.append(df_k)
df_metrics = pd.concat(metrics)
df_guess = pd.concat(best_k)

df_guess['hit'] = np.int_(df_guess['guess'].values == 6)
df_guess['abshit'] = np.abs(df_guess['guess'].values - 6)
df_great_mean = apply_function(df_guess, keys=['name', 'algorithm'], attr='abshit', fx=np.mean)
df_great_mean = apply_function(df_guess, keys=['name', 'algorithm'], attr='hit', fx=np.mean)

# Plot of metrics
df_mean = apply_function(df_guess, keys=['name'], attr='hit', fx=np.mean)
arg_sort = np.argsort(df_mean['hit'].values)[::-1]

for alg in np.unique(df_great_mean['algorithm']):
    df_a = filter_dataframe(df_great_mean, algorithm=[alg])
    values = df_a['hit'].values[arg_sort]
    pl.plot(values, '-o')
pl.xticks(np.arange(len(values)), df_a['name'].values[arg_sort])
    
# State similarity
df = calculate_centroids(df)
df = state_errors(df)
df = dynamics_errors(df)


#################################

conditions = {

    'time': [0.5, 1, 1.5, 2.],
    #'num': [str(j) for j in np.arange(1, 480)],
    'snr': [10, 100, 1000, 10000],
    #'algorithm': ['GaussianMixture',
    #              'KMeans',
    #              'AgglomerativeClustering', 
    #              'SpectralClustering',
    #              'MiniBatchKMeans'],
    'subject': [str(i) for i in np.arange(1, 21)]
    
}

combinations = make_dict_product(**conditions)

metrics = []
best_k = []

algorithms = ['GaussianMixture',
             'KMeans',
             'AgglomerativeClustering', 
             'SpectralClustering',
             'MiniBatchKMeans']

for algorithm in algorithms:

    filter = {"algorithm":[algorithm]}
    if algorithm == 'GaussianMixture':
        filter.update({"n_components": ['6']})
    else:
        filter.update({"n_states": ['6']})


    data = get_results('/media/robbis/Seagate_Pt1/data/simulations/derivatives/', 
                   pipeline='c2b+real', 
                   field_list=['sample_slicer', 
                               'n_clusters', 
                               'n_components', 
                               'ds.a.snr',
                               'ds.a.time',
                               'ds.a.states',
                               'fetch', 
                               'algorithm'],
                    n_jobs=5,
                    filter=filter)
    
    df = purge_dataframe(data, keys=['ds.a.snr', 
                                'ds.a.time', 
                                #'ds.a.states',
                                'n_clusters', 
                                'n_components'])

    df.to_hdf("/media/robbis/Seagate_Pt1/data/simulations/derivatives/data_%s_n_6.hd5", 
              "data")

    for options in tqdm(combinations):
        df_ = filter_dataframe(df, **options)
        options = {k: v[0] for k, v in options.items()}
        options.update({'algorithm':algorithm})
        df_metric = calculate_metrics(df_, fixed_variables=options)
        df_metric = df_metric.sort_values('k')
        df_k = find_best_k(df_metric)
        metrics.append(df_metric)
        best_k.append(df_k)
    
    del data, df
    gc.collect()

df_metrics = pd.concat(metrics)
df_guess = pd.concat(best_k)

df_guess['hit'] = np.int_(df_guess['guess'].values == 6)
df_guess['abshit'] = np.abs(df_guess['guess'].values - 6)
df_great_mean = apply_function(df_guess, keys=['name', 'algorithm'], attr='abshit', fx=np.mean)
df_great_mean = apply_function(df_guess, keys=['name', 'algorithm'], attr='hit', fx=np.mean)

##### Plot of hits by algorithm #####
df_guess = pd.read_csv("/home/robbis/Dropbox/simulation_guess.csv")
_, mask = filter_dataframe(df_guess, return_mask=True, algorithm=['GaussianMixture'])
df_guess = df_guess.loc[np.logical_not(mask)]
df_great_mean = apply_function(df_guess, keys=['name', 'algorithm', 'snr', 'time'], attr='hit', fx=np.mean)

df_sort = apply_function(df_guess, keys=['name'], attr='hit', fx=np.mean).sort_values(by='hit')
encoding = dict(zip(df_sort['name'].values, np.arange(7)[::-1]))
df_great_mean['metric'] = [encoding[name] for name in df_great_mean['name'].values]
df_guess['metric'] = [encoding[name] for name in df_guess['name'].values]

xlabels = list(encoding.keys())[::-1]
xlabels = ['SIL', 'GEV', 'WGSS', "CV", "EV", "KL", "I"]
#### Totale #####
palette = sns.color_palette("magma", 8)[::-1][::2]

f = sns.relplot(x="metric", y="hit", row="time", col="algorithm", hue="snr", data=df_great_mean, 
                kind='line', height=6, aspect=.75, palette=palette,
                legend="full", marker='o', lw=3.5, markersize=15, markeredgecolor='none')
for ax in f.axes[-1]:
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)

### Best metric ###
f = sns.relplot(x="metric", y="hit", hue="algorithm", data=df_guess,
            kind='line', marker='o', 
            lw=3.5, markersize=15, markeredgecolor='none')
for ax in f.axes[-1]:
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)

### Metric vs snr ###
palette = sns.color_palette("magma", 8)[::-1][::2]
f = sns.relplot(x="metric", y="hit", hue="snr", data=df_guess, palette=palette,
            kind='line', marker='o', 
            lw=3.5, markersize=15, markeredgecolor='none')
for ax in f.axes[-1]:
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)

### Metric vs time ###
df_sort = apply_function(df_guess, keys=['name', 'snr'], attr='hit', fx=np.mean).sort_values(by='hit')
df_sort['metric'] = [encoding[name] for name in df_sort['name'].values]
palette = sns.color_palette("magma", 8)[::-1][::2]
f = sns.relplot(x="metric", y="hit", hue="time", data=df_guess, palette=palette,
            kind='line', marker='o', lw=3.5, markersize=15, markeredgecolor='none')
for ax in f.axes[-1]:
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)         

###########################################################à
algorithms = ['GaussianMixture',
             'KMeans',
             'AgglomerativeClustering', 
             'SpectralClustering',
             'MiniBatchKMeans']

full_df = []

for algorithm in algorithms:

    filter = {"algorithm":[algorithm]}
    if algorithm == 'GaussianMixture':
        filter.update({"n_components": ['6']})
    else:
        filter.update({"n_clusters": ['6']})


    data = get_results('/media/robbis/Seagate_Pt1/data/simulations/derivatives/', 
                   pipeline='c2b+real', 
                   field_list=['sample_slicer', 
                               'n_clusters', 
                               'n_components', 
                               'ds.a.snr',
                               'ds.a.time',
                               'ds.a.states',
                               'fetch', 
                               'algorithm'],
                    n_jobs=5,
                    filter=filter)
    
    df = purge_dataframe(data, keys=['ds.a.snr', 
                                'ds.a.time', 
                                'ds.a.states',
                                'n_clusters', 
                                'n_components'])
    df = calculate_centroids(df)
    df = state_errors(df)
    df = dynamics_errors(df)
    f = sns.relplot(x="time", y="dynamics_errors", hue="snr", 
         height=5, aspect=.75, facet_kws=dict(sharex=False), 
         kind="line", legend="full", data=df, palette=['r','g','b', 'orange'], marker='o')
    f = sns.relplot(x="time", y="centroid_similarity", hue="snr", 
         height=5, aspect=.75, facet_kws=dict(sharex=False), 
         kind="line", legend="full", data=df, palette=[], marker='o')


    full_df.append(df)


full_df = pd.concat(full_df)

############################

palette = sns.color_palette("magma", 8)[::-1]
f = sns.relplot(x="time", y="dynamics_errors", hue="snr", col="algorithm",
        height=5, aspect=.75, facet_kws=dict(sharex=False), 
        kind="line", legend="full", data=full_df, palette=palette[::2], 
        marker='o', lw=3.5, markersize=15, markeredgecolor='none')

f = sns.relplot(x="time", y="centroid_similarity", hue="snr", col="algorithm",
        height=5, aspect=.75, facet_kws=dict(sharex=False), 
        kind="line", legend="full", data=full_df, palette=palette[::2], 
        marker='o', lw=3.5, markersize=15, markeredgecolor='none')

flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
palette = sns.color_palette(flatui)
f = sns.relplot(x="time", y="dynamics_errors", col="snr", hue="algorithm",
        height=5, aspect=.75, facet_kws=dict(sharex=False), 
        kind="line", legend="full", data=full_df, palette=palette, 
        marker='o', lw=3.5, markersize=15, markeredgecolor='none')

f = sns.relplot(x="time", y="centroid_similarity", col="snr", hue="algorithm",
        height=5, aspect=.75, facet_kws=dict(sharex=False), 
        kind="line", legend="full", data=full_df, palette=palette, 
        marker='o', lw=3.5, markersize=15, markeredgecolor='none')


########################################

from pyitab.plot.connectivity import plot_connectivity_lines
from matplotlib import animation
from mvpa2.base.hdf5 import h5load

path = "/home/robbis/mount/aalto-work/data/simulations/meg/ds-min_time_1.5-snr_10000.gzip"
ds = h5load(path)

samples = ds.samples
matrices = np.array([copy_matrix(array_to_matrix(m)) for m in samples[::50]])



names = ["node_%02d"%(i+1) for i in range(10)]

def animate(i, fig):
    names = ["node_%s" % (str(j+1)) for j in range(10)]
    #pl.imshow(matrix[i*100])
    pl.clf()
    plot_connectivity_lines(matrices[i], facecolor='white',
                            node_names=names, con_thresh=0., 
                            kind='circle', fig=fig)


fig = pl.figure(figsize=(8, 8))

anim = animation.FuncAnimation(fig, animate, fargs=[fig],
                               frames=45, interval=20)
anim.save('/home/robbis/animation.gif', writer='imagemagick', fps=10)