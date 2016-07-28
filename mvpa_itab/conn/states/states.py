import numpy as np
import scipy

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.io import loadmat, savemat

from nitime.timeseries import TimeSeries
from mvpa_itab.similarity.analysis import SeedAnalyzer
from mvpa_itab.conn.states.states_metrics import *

from scipy.signal import argrelextrema, argrelmin


def get_data(filename):
    """
    Returns upper triangular matrix data and number of ROIs
    """
    
    #filename = '/media/robbis/DATA/fmri/movie_viviana/mat_corr_sub_REST.mat'
    data = loadmat(filename)
    data = np.array(data['data'], dtype=np.float16)
    ix, iy = np.triu_indices(data.shape[-1], k=1)
    data = data[:,:,ix,iy]
    
    return data, data.shape[-1]


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
    return np.array([X[labels == l].mean(0) for l in np.unique(labels)])


def fit_states(X, centroids, distance=euclidean):
    

    ts_seed = TimeSeries(centroids, sampling_interval=1.)
    
    results_ = []
    
    for subj in X:
        ts_target = TimeSeries(subj, sampling_interval=1.)
        S = SeedAnalyzer(ts_seed, ts_target, euclidean)
        results_.append(S.measure)
        
    
    return results_


def calculate_metrics(X, 
                      clustering_labels, 
                      metrics={'Silhouette': metrics.silhouette_score,
                               'Calinski-Harabasz': ch_criterion, 
                               'Krzanowski-Lai': kl_criterion,
                               'Explained Variance':explained_variance,
                               'Gap': gap,
                               }):
    
    metrics_ = []
    k_step = np.zeros(len(clustering_labels), dtype=np.int8)
    
    for i, label in enumerate(clustering_labels):
        
        k = len(np.unique(label))
        k_step[i] = k
        
        print '----- '+str(k)+' -------'
        
        metric_list = []
        
        for metric_name, metric_function in metrics.items():
            
            if metric_name == 'Krzanowski-Lai':
                if i == 0 or i == len(clustering_labels) - 1:
                    prev_labels = None
                    next_labels = None
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
    
    
    return np.array(metrics_), k_step, metrics.keys()

