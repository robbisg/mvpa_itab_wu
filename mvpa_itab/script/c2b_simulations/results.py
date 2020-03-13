from pyitab.analysis.results.simulations import get_results, purge_dataframe, \
    calculate_metrics, find_best_k, calculate_centroids, state_errors, \
    dynamics_errors
from pyitab.analysis.results.base import filter_dataframe
from pyitab.analysis.results.dataframe import apply_function
from pyitab.utils import make_dict_product
import pandas as pd
import numpy as np

data = get_results('/u/97/guidotr1/unix/data/simulations/meg/derivatives', 
                   pipeline='c2b+real', 
                   field_list=['sample_slicer', 
                               'n_clusters', 
                               'n_components', 
                               'ds.a.snr',
                               'ds.a.time',
                               'ds.a.states',
                               'fetch', 
                               'algorithm'])

df = purge_dataframe(data)




conditions = {

    #'time': [0.5, 1, 1.5, 2.],
    #'num': [str(j) for j in np.arange(1, 480)],
    #'snr': [10, 100, 1000, 10000],
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






from pyitab.plot.connectivity import plot_connectivity_lines
from matplotlib import animation

def animate(i, fig):
    names = ["node_%s" % (str(i+1)) for i in range(10)]
    pl.imshow(matrix[i*100])
    #plot_connectivity_lines(matrix[i*50], node_names=names, con_thresh=0.3, kind='multi', fig=fig)


fig = pl.figure(figsize=(10, 10), 
                        facecolor='black')

anim = animation.FuncAnimation(fig, animate, fargs=[fig], 
                               frames=500, interval=20)



df_silhouette = filter_dataframe(df_guess, name=['Silhouette'])

grid = sns.FacetGrid(df_silhouette, row='algorithm', hue='snr', palette="tab20c")
grid.map(pl.plot, "time", "hit", marker='o')



df_mean = apply_function(df_metrics, keys=['name', 'k', 'time', 'snr', 'algorithm'], attr='value', fx=np.mean)
for metric in np.unique(df_metrics['name'].values):
    #pl.figure()
    df_filt = filter_dataframe(df_mean, name=[metric])
    print(metric)
    grid = sns.FacetGrid(df_filt, col="algorithm", row='snr', hue='time', palette="tab20c")
    grid.map(pl.plot, "k", "value", marker="o")