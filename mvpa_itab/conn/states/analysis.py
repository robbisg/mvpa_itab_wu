from mvpa_itab.conn.states.base import cluster_state
from mvpa_itab.conn.states.utils import get_data, get_centroids
from mvpa_itab.conn.states.metrics import calculate_metrics
import os
import cPickle as pickle

def demeaned():
    
    conf = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
            'filetype' : 'masked',
            'fname': 'mat_corr_sub_%s.mat',
            'conditions' : ['movie', 'scramble', 'rest'],
            'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
            'max_k':15,
            'method':'speed'                 
            }
 
    filename = os.path.join(conf['path'], conf['filetype'], conf['fname'])
    
    for condition in conf["conditions"]:
        data, n_roi = get_data(filename % (str(condition)))
        
        
        #data = remove_component(data, method="mean", argument="session")
        data = data - data.mean(1)[:,None,:]

        X, clustering_ = cluster_state(data, range(2,8), method='variance')
        metrics, k_step, metric_names = calculate_metrics(X, clustering_)
        
        
        
        clustering_ = pickle.load(file(label_fname % (condition.lower()), 'r'))
        
        centroid_ = get_centroids(X, clustering_[3]) # Five centroids
        centroid_ = centroid_[order[condition],:] # Similarity reorder
        
        dict_centroids[condition.lower()] = centroid_
    get_positions(dict_centroids, path)
    



    