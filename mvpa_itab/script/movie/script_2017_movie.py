import cPickle as pickle
import os
import numpy as np
import itertools
from sklearn.manifold.mds import MDS
from nitime.timeseries import TimeSeries
from mvpa_itab.similarity.analysis import SimilarityAnalyzer
from mvpa_itab.conn.states.base import get_subsampling_measure
from mvpa_itab.conn.states.pipelines import get_centers
from mvpa_itab.conn.states.utils import get_data
from mvpa_itab.conn.states.metrics import calculate_metrics
from mvpa_itab.conn.states.plot import plot_metrics

"""
distance_fname = os.path.join('/media/robbis/DATA/fmri/movie_viviana/',
                              'pairwise_distance_all_subj_all_pts_movie.npy')
clustering_ = pickle.load(file('/media/robbis/DATA/fmri/movie_viviana/clustering_labels_2to30k_movie_all_sub_all_pts.obj', 'r'))
distance_matrix = np.load(distance_fname, mmap_mode='r')
"""


configuration = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
                 'filetype' : 'masked',
                 'fname': 'mat_corr_sub_%s.mat',
                 'conditions' : ['movie', 'scramble', 'rest'],
                 'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
                 'max_k':15,
                 'method':'speed'                 
                 }


filetypes = ['masked']
method = ['speed', 'variance']
conditions = ['movie', 'scramble', 'rest']

X_centers = dict()
prod = itertools.product(filetypes, method, conditions)


for t, m, c in prod:
    path_cluster = "%s/%s/%s" % (t, m, c)
    path_cluster = os.path.join(configuration['path'], path_cluster)
    fname = configuration['state_res_fname'] % (c, 
                                                str(10), 
                                                m, 
                                                t)
    
    
    path_file = os.path.join(path_cluster, fname)
    clustering_ = pickle.load(file(path_file, 'r'))
    
    data_fname = os.path.join(configuration['path'], t, configuration['fname'])
    data_, _ = get_data(data_fname % str.upper(c))
    subj_min_speed, subj_speed = get_subsampling_measure(m)(data_)

    X = data_[subj_min_speed]   
    
    metrics_, k_step, metric_names = calculate_metrics(X, clustering_)
    
    fig = plot_metrics(metrics_, metric_names, k_step)
    fig_fname = os.path.join(configuration['path'], t, m, m+"_metric_evaluation.png")
    
    fig.savefig(fig_fname)
    
    #####################

    
centers, conditions = get_centers( n_cluster=5, method='variance', filetype='masked', max_k=10)
n_conditions = len(conditions)

centroid_ts = TimeSeries(np.vstack(centers), sampling_interval=1)
S = SimilarityAnalyzer(centroid_ts)

distance_ = S.measure

movie_dist = distance_[:5,5:]
colors = ['blue', 'green']


fig = pl.figure()
for j in range(5):
    for i in range(2):
        ax = fig.add_subplot(5,1,j+1)
        ax.plot(np.arange(1,6),
                movie_dist[j,5*(i):5*(i+1)], 
                marker='o', lw=2.5, 
                markersize=10, 
                c=colors[i])
        ax.set_xticks(np.arange(1,6))
        ax.set_xticklabels(np.arange(1,6))
        ax.set_ylabel("Movie state %s" % (str(j+1)))
    
    
for i in range(10):
    pos = MDS(n_components=2).fit_transform(np.vstack(centers))
    pl.figure()
    pl.scatter(pos[:,0], pos[:,1], c=color)
    
    