import numpy as np
import scipy
import os
import cPickle as pickle

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.io import loadmat, savemat

from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer
from mvpa_itab.similarity.analysis import SeedAnalyzer
from mvpa_itab.conn.states.states_metrics import *

from scipy.signal import argrelextrema, argrelmin



def get_data(filename):
    """
    Returns upper triangular matrix data and number of ROIs.
    The data is specific for the task, is a non-stationary connectivity
    matrix. 
    
    Parameters
    ----------
    filename : string
        The filename of a .mat matlab file in which the matrix is
        a data variable in matlab.
        
    Returns
    -------
    data : n_session x n_timepoints x n_connections numpy array
        The upper-part of the non-stationary matrix
        
    n_roi : int
        Number of ROI of the original matrix.
    """
    
    #filename = '/media/robbis/DATA/fmri/movie_viviana/mat_corr_sub_REST.mat'
    data = loadmat(filename)
    data = np.array(data['data'], dtype=np.float16)
    n_roi = data.shape[-1]
    ix, iy = np.triu_indices(data.shape[-1], k=1)
    data = data[:,:,ix,iy]
    
    return data, n_roi


def get_max_variance_arguments(data):
    
    stdev_data = data.std(axis=2)
    arg_maxima = argrelextrema(stdev_data, np.greater, axis=1)
    
    return arg_maxima, stdev_data
    


def get_min_speed_arguments(data):
    
    subj_speed = []
    for i in range(data.shape[0]):
        distance_ = squareform(pdist(data[i], 'euclidean'))
        
        speed_ = [distance_[i, i+1] for i in range(distance_.shape[0]-1)]
        subj_speed.append(np.array(speed_))
    
    subj_speed = np.vstack(subj_speed)
    subj_min_speed = argrelmin(np.array(subj_speed), axis=1)
    
    return subj_min_speed, subj_speed
    


def get_centroids(X, labels):
    """
    Returns the centroid of a clustering experiment
    
    Parameters
    ----------
    X : n_samples x n_features array
        The full dataset used for clustering
    
    labels : n_samples array
        The clustering labels for each sample.
        
        
    Returns
    -------
    centroids : n_cluster x n_features shaped array
        The centroids of the clusters.
    """
    
    return np.array([X[labels == l].mean(0) for l in np.unique(labels)])


def fit_centroids(X, centroids):
    
    k = centroids.shape[0]
    
    results_ = []
    
    for subj in X:
        km = KMeans(n_clusters=k,
                    init=centroids).fit(subj)
                    
        results_.append(km.labels_)
        
    return np.array(results_)




def fit_states(X, centroids, distance=euclidean):
    """
    Returns the similarity of the dataset to each centroid,
    using a dissimilarity distance function.
    
    Parameters
    ----------
    X : n_samples x n_features array
        The full dataset used for clustering
    
    centroids : n_cluster x n_features array
        The cluster centroids.
        
    distance : a scipy.spatial.distance function | default: euclidean
        This is the dissimilarity measure, this should be a python
        function, see scipy.spatial.distance.
        
    
    Returns
    -------
    results : n_samples x n_centroids array
        The result of the analysis,
    
    """
    

    ts_seed = TimeSeries(centroids, sampling_interval=1.)
    
    results_ = []
    
    for subj in X:
        ts_target = TimeSeries(subj, sampling_interval=1.)
        S = SeedAnalyzer(ts_seed, ts_target, distance)
        results_.append(S.measure)
        
    
    return results_



def get_state_frequencies(state_dynamics, method='spectrum_fourier'):
    """
    Returns the spectrum of the state occurence for each subject.
    
    Parameters
    ----------
    state_dynamics :    n_states x n_subjects x n_timepoints array
                        The state dynamics output from fit_states
                        function.
                        
    method : a string, check nitime.spectral.SpectralAnalyzer for 
             allowed methods.
    
    
    Returns
    -------
    results : n_subjects list of tuple,
              first element is the array of frequencies,
              second element is the array n_states x frequencies
              of the spectrum.
    
    """
    
    results = []
    for s in state_dynamics:
        ts = TimeSeries(s, sampling_interval=1.)
        S = SpectralAnalyzer(ts)
        try:
            result = getattr(S, method)
        except AttributeError, err:
            result = S.spectrum_fourier
        
        results.append(result)
        
    return results



def get_state_probability():
    return



def calculate_metrics(X, 
                      clustering_labels, 
                      metrics_kwargs=None):
    
    default_metrics = {'Silhouette': metrics.silhouette_score,
                       'Calinski-Harabasz': ch_criterion, 
                       'Krzanowski-Lai': kl_criterion,
                       #'Explained Variance':explained_variance,
                       #'Gap': gap,
                        }
    if metrics_kwargs != None:
        default_metrics.update(metrics_kwargs)
    
    metrics_ = []
    k_step = np.zeros(len(clustering_labels), dtype=np.int8)
    
    for i, label in enumerate(clustering_labels):
        
        k = len(np.unique(label))
        k_step[i] = k
        
        print '----- '+str(k)+' -------'
        
        metric_list = []
        
        for metric_name, metric_function in default_metrics.items():
            if metric_name == 'Krzanowski-Lai':
                if i == len(clustering_labels) - 1:
                    prev_labels = clustering_labels[i-1]
                    next_labels = np.arange(0, label.shape[0])
                elif k == 2:
                    prev_labels = np.zeros_like(label)
                    next_labels = clustering_labels[i+1]
                else:   
                    prev_labels = clustering_labels[i-1]
                    next_labels = clustering_labels[i+1]
                    
                m_ = metric_function(X,
                                     label,
                                     previous_labels=prev_labels,
                                     next_labels=next_labels,
                                     precomputed=False)
                
            else:
                m_ = metric_function(X, label)
                
            
            metric_list.append(m_)
            
        
        metrics_.append(metric_list)
    
    
    return np.array(metrics_), k_step, default_metrics.keys()



def cluster_state(data, max_k=30, method='speed'):
    
    """
    if method != 'speed':
    ### Variance ###
        arg_maxima, stdev_data = get_max_variance_arguments(data)
        hist_arg = get_extrema_histogram(arg_maxima, data.shape[1])
        X = data[arg_maxima]
    else:
    ### Speed ###
        subj_min_speed, subj_speed = get_min_speed_arguments(data)
        hist_arg = get_extrema_histogram(subj_min_speed, data.shape[1])
        X = data[subj_min_speed]
    """
    
    method = get_subsampling_method(method)
    arg_, sub_data = method(data)
    hist_arg = get_extrema_histogram(arg_, data.shape[1])
    X = data[arg_]
    
    clustering_ = []
    
    k_steps = range(2,max_k)
    
    for k in k_steps:
        print '----- '+str(k)+' -------'
        km = KMeans(n_clusters=k).fit(X)
        labels = km.labels_
        clustering_.append(labels)
        
    return X, clustering_



def get_extrema_histogram(arg_extrema, n_timepoints):
    
    hist_arg = np.zeros(n_timepoints)
    n_subjects = len(np.unique(arg_extrema[0]))
    
    for i in range(n_subjects):
        sub_max_arg = arg_extrema[1][arg_extrema[0] == i]
        hist_arg[sub_max_arg] += 1
        
    return hist_arg


def get_subsampling_method(method):
    
    method_mapping = {
                      'speed': get_min_speed_arguments,
                      'variance': get_max_variance_arguments
                      }
    
    
    return method_mapping[method]



def analysis(**kwargs):
    """
    Method to fast analyze states from matrices.
  
    
    Parameters
    ----------
    kwargs : dictionary of several parameters
    
        path : string of the data path
        filetype : string ('masked', 'original')
                if you want to use masked matrix or full rank.
        fname : string. pattern of the mat file in input
            default ("mat_corr_sub_%s.mat")
        conditions : list of string of data conditions. 
                should cope with file pattern specified in fname.
        method : string ('speed', 'variance')
                Method used to subsample data.
        max_k : integer (default = 15).
                The maximum number of cluster to use
        state_res_fname : pattern of the output file 
            (default "clustering_labels_%s_maxk_%s_%s_%s.pyobj")
                File used to save labels after clustering.
    
    
    """
    from mvpa_itab.conn.states.utils import plot_metrics, plot_states_matrices
    
    configuration = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
                     'filetype' : 'masked',
                     'fname': 'mat_corr_sub_%s.mat',
                     'conditions' : ['movie', 'scramble', 'rest'],
                     'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
                     'max_k':15,
                     'method':'speed'                 
                     }
    
    configuration.update(kwargs)
    
    conditions = configuration['conditions']
    max_k = configuration['max_k']
    method = configuration['method']
    filetype = configuration['filetype']
    
    
    for cond in conditions:
    
        path = os.path.join(configuration['path'], configuration['filetype'])
        
        data_, n_roi = get_data(os.path.join(path,"mat_corr_sub_%s.mat" % (str.upper(cond))))
        
        X, clustering_ = cluster_state(data_, max_k, method)
        
        metrics_, k_step, metrics_keys = calculate_metrics(X, clustering_)
        
        
        fig = plot_metrics(metrics_, metrics_keys, k_step)
        
        
        directory = os.path.join(path, method, cond)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        state_res_fname = configuration['state_res_fname']
        pickle.dump(clustering_, file(os.path.join(directory, 
                                                   state_res_fname %(cond,
                                                                    str(max_k),
                                                                    method,
                                                                    filetype), 
                                                       ),
                                      'w'))
    
        fig_fname = os.path.join(directory, "metrics.png")
        fig.savefig(fig_fname)
        
        for i, cl in enumerate(clustering_):
            if not os.path.exists(os.path.join(directory,str(i+2))):
                os.makedirs(os.path.join(directory,str(i+2)))
            plot_states_matrices(X, 
                                 cl,
                                 os.path.join(directory,str(i+2)),
                                 cond)
        
        
        
        
        
        
        