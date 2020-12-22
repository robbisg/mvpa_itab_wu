from scipy.io import loadmat, savemat
from scipy.signal import argrelextrema
from mvpa_itab.conn.operations import copy_matrix, array_to_matrix
from scipy.spatial.distance import squareform, pdist, euclidean, correlation
from scipy.signal._peak_finding import argrelmin
from mvpa_itab.conn.states.plot import plot_dynamics, plot_frequencies, plot_metrics, \
                                    plot_states_matrices, get_positions
from mvpa_itab.conn.states.utils import get_centroids, fit_states, cluster_state, \
                                calculate_metrics, get_data, get_min_speed_arguments, \
                                get_extrema_histogram, fit_centroids,\
                                get_subsampling_measure

import cPickle as pickle
import os
import numpy as np
import itertools
from sklearn.manifold.mds import MDS

"""
distance_fname = os.path.join('/media/robbis/DATA/fmri/movie_viviana/',
                              'pairwise_distance_all_subj_all_pts_movie.npy')
clustering_ = pickle.load(file('/media/robbis/DATA/fmri/movie_viviana/clustering_labels_2to30k_movie_all_sub_all_pts.obj', 'r'))
distance_matrix = np.load(distance_fname, mmap_mode='r')
"""

path = "/media/robbis/DATA/fmri/movie_viviana/masked/"
conditions = ['rest', 'movie', 'scramble']
filetype = 'masked'

for cond in conditions:
    data_, n_roi = get_data(os.path.join(path,"mat_corr_sub_%s_masked.mat" % (str.upper(cond))))
    
    max_k = 15
    method = 'speed'
    
    X, clustering_ = cluster_state(data_, max_k, method)
    
    metrics_, k_step, metrics_keys = calculate_metrics(X, clustering_)
    pickle.dump(clustering_, file(os.path.join(path, 
                                               "clustering_labels_%s_maxk_%s_%s_%s.pyobj" %(cond,
                                                                                            str(max_k),
                                                                                            method,
                                                                                            filetype), 
                                                   ),
                                  'w'))
    
    
    plot_states_matrices(X, 
                         clustering_[3],
                         path,
                         cond)

#################

order = {'MOVIE': [0,1,2,3,4],
         'REST': [4,3,1,0,2],
         'SCRAMBLE' : [1,2,4,0,3]
         }

order = {'MOVIE': [0,1,2,3,4],
         'REST': [3,1,2,4,0],
         'SCRAMBLE' : [0,4,2,1,3]
         }

path = '/media/robbis/DATA/fmri/movie_viviana/masked'
fname = '/media/robbis/DATA/fmri/movie_viviana/masked/mat_corr_sub_%s_masked.mat'
label_fname = '/media/robbis/DATA/fmri/movie_viviana/clustering_labels_2to30k_%s_all_sub_speed.obj'
label_fname = os.path.join(path,"clustering_labels_%s_maxk_15_speed_masked.pyobj")
dict_centroids = dict()

conditions = ['MOVIE', 'SCRAMBLE', 'REST']

for condition in conditions:
    data_, n_roi = get_data(fname % (condition))
    subj_min_speed, subj_speed = get_min_speed_arguments(data_)
    hist_arg = get_extrema_histogram(subj_min_speed, data_.shape[1])
    X = data_[subj_min_speed] 
    
    
    clustering_ = pickle.load(file(label_fname % (condition.lower()), 'r'))
    
    centroid_ = get_centroids(X, clustering_[3]) # Five centroids
    centroid_ = centroid_[order[condition],:] # Similarity reorder
    
    dict_centroids[condition.lower()] = centroid_
    
    state_dynamics = fit_states(data_, centroid_, distance=euclidean)
    state_dynamics = np.array(state_dynamics)
    state_dynamics = np.nan_to_num(state_dynamics)
    #state_frequency = get_state_frequencies(state_dynamics)
    
    plot_dynamics(state_dynamics, condition, path)
    #plot_frequencies(state_frequency, condition, path)
    
    state_centroids = fit_centroids(data_, centroid_)
    state_centroids = state_centroids[:,np.newaxis,:]+1
    plot_dynamics(state_centroids, condition, path, suffix='_kmeans')
    
    plot_states_matrices(centroid_, 
                         clustering_[3], 
                         path, 
                         condition.lower(),
                         use_centroid=True)
    
    

###################################################


order = {'MOVIE': [0,1,2,3,4],
         'REST': [4,3,1,0,2],
         'SCRAMBLE' : [1,2,4,0,3]
         }
'''
order = {'MOVIE': [0,1,2,3,4],
         'REST': [3,1,2,4,0],
         'SCRAMBLE' : [0,4,2,1,3]
         }
'''
#path = '/media/robbis/DATA/fmri/movie_viviana/masked'
path = '/media/robbis/DATA/fmri/movie_viviana/original'
#fname = '/media/robbis/DATA/fmri/movie_viviana/masked/mat_corr_sub_%s_masked.mat'
fname = '/media/robbis/DATA/fmri/movie_viviana/original/mat_corr_sub_%s.mat'
label_fname = '/media/robbis/DATA/fmri/movie_viviana/original/clustering_labels_2to30k_%s_all_sub_speed.obj'
#label_fname = os.path.join(path,"clustering_labels_%s_maxk_15_speed_masked.pyobj")
dict_centroids = dict()

conditions = ['MOVIE', 'SCRAMBLE', 'REST']

for condition in conditions:
    data_, n_roi = get_data(fname % (condition))
    subj_min_speed, subj_speed = get_min_speed_arguments(data_)
    hist_arg = get_extrema_histogram(subj_min_speed, data_.shape[1])
    X = data_[subj_min_speed] 
    
    clustering_ = pickle.load(file(label_fname % (condition.lower()), 'r'))
    
    centroid_ = get_centroids(X, clustering_[3]) # Five centroids
    centroid_ = centroid_[order[condition],:] # Similarity reorder
    
    dict_centroids[condition.lower()] = centroid_
get_positions(dict_centroids, path)
    

###############################################################

configuration = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
                 'filetype' : 'masked',
                 'fname': 'mat_corr_sub_%s.mat',
                 'conditions' : ['movie', 'scramble', 'rest'],
                 'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
                 'max_k':15,
                 'method':'speed'                 
                 }


filetypes = ['masked']
method = ['variance']
conditions = ['movie', 'scramble', 'rest']

X_centers = dict()
prod = itertools.product(filetypes, method, conditions)
max_k = 10

for p in prod:
    
    data_fname = os.path.join(configuration['path'], p[0], configuration['fname'])
    data_, _ = get_data(data_fname % str.upper(p[2]))
    subj_min_speed, subj_speed = get_subsampling_measure(p[1])(data_)

    X = data_[subj_min_speed]
        
    path_cluster = "%s/%s/%s" % (p[0], p[1], p[2])
    path_cluster = os.path.join(configuration['path'], path_cluster)
    
    fname = configuration['state_res_fname'] % (p[2], 
                                                str(max_k), 
                                                p[1], 
                                                p[0])
    
    
    path_file = os.path.join(path_cluster, fname)
    X_centers[path_cluster] = []
    clustering_ = pickle.load(file(path_file, 'r'))
    
    for i in [5]:#np.arange(max_k):

        X_centers[path_cluster].append(get_centroids(X, clustering_[i-2]))
        
    
##########################################################

def sum_index(k):
    return int(np.sum([n for n in np.arange(2,k)]))

filetypes = ['masked']
method = ['variance']
conditions = ['movie', 'scramble', 'rest']
prod = itertools.product(filetypes, method)

color = np.zeros((3*104, 1))
color[104:2*104] = 10
color[2*104:3*104] = 50

for p in prod:
    XX = []
    for c in conditions:
        print c
        path_cluster = "%s/%s/%s" % (p[0], p[1], c)
        path_cluster = os.path.join(configuration['path'], path_cluster)
        
        XX.append(np.vstack(X_centers[path_cluster]))
    
    XX = np.array(XX)
    
    for n_cluster in [5]:#np.arange(2,7):
        
        index1 = sum_index(n_cluster)
        index2 = sum_index(n_cluster+1)
        
        X_k = XX[:,index1:index2,:]
        assert n_cluster == X_k.shape[1]
    
        mds = MDS(n_components=2)        
        pos = mds.fit_transform(np.vstack(X_k))
        
        color = np.zeros((pos[:,0].shape[0], 3))
        color[:X_k.shape[1],0] = 1 # Red - Movie
        color[2*X_k.shape[1]:,1] = 1 # Green - Rest
        color[X_k.shape[1]:2*X_k.shape[1],2] = 1 # Blue - Scramble
        
        pl.figure(figsize=(10,8))
        pl.scatter(pos[:,0], pos[:,1], c=color, s=120)
        for i, (x,y) in enumerate(pos):
            pl.annotate(int(i%n_cluster)+1, (x+0.1, y+.1))
        figname = "mds_%02d_new.png" % (n_cluster)
        pl.savefig(os.path.join(configuration['path'], p[0], p[1], figname))
        
    