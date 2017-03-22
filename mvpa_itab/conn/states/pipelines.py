import os
from mvpa_itab.conn.states.utils import get_data, get_centroids, save_data
from mvpa_itab.conn.states.metrics import calculate_metrics
from mvpa_itab.conn.states.base import cluster_state

import cPickle as pickle
from mvpa_itab.conn.states.filters import get_filter
from mvpa_itab.conn.states.subsamplers import get_subsampler
import logging
from scipy.io.matlab.mio import loadmat

logger = logging.getLogger(__name__)

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
        method : string ('speed', 'variance', 'variance+mean', 'speed+mean')
                Method used to subsample data. Using '+mean' suffix let 
                to select only points that are greater than mean.
        max_k : integer (default = 15).
                The maximum number of cluster to use
        filter : string ('none', 'normalize')
                Method used to normalize and filter data.
        state_res_fname : pattern of the output file 
            (default "clustering_labels_%s_maxk_%s_%s_%s.pyobj")
                File used to save labels after clustering.
    
    
    """
    from mvpa_itab.conn.states.plot import plot_metrics, plot_states_matrices
    
    configuration = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
                     'band': 'alpha',
                     'filetype' : 'masked',
                     'fname': 'mat_corr_sub_%s.mat',
                     'conditions' : ['movie', 'scramble', 'rest'],
                     'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
                     'max_k':15,
                     'method':'variance+mean',
                     'filter':'none'                
                     }
    
    configuration.update(kwargs)
    
    logger.info(configuration)
    
    conditions = configuration['conditions']
    max_k = configuration['max_k']
    method = configuration['method']
    filetype = configuration['filetype']
    
    filter_type = configuration['filter']
    data_filter = get_filter(configuration['filter'])
    
    result = dict()
    
    for cond in conditions:
        
    
        path = os.path.join(configuration['path'], 
                            configuration['band'], 
                            configuration['filetype'])
        
        data_, n_roi = get_data(os.path.join(path, "mat_corr_sub_%s.mat" % (str.upper(cond))))
        
        data_ = data_filter.fit(data_)
            
        subsampler = get_subsampler(method)
        
        X = subsampler.fit(data_).subsample(data_)
        
        X, clustering_ = cluster_state(X, range(2, max_k))
        
        metrics_, k_step, metrics_keys = calculate_metrics(X, clustering_)
        
        
        fig = plot_metrics(metrics_, metrics_keys, k_step)
        
        
        directory = os.path.join(path, method, filter_type, cond)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save results
        state_res_fname = configuration['state_res_fname']
        pickle.dump(clustering_, file(os.path.join(directory, 
                                                   state_res_fname %(cond,
                                                                    str(max_k),
                                                                    method,
                                                                    filetype), 
                                                       ),
                                      'w'))
        
        # Save metrics
        fig_fname = os.path.join(directory, "metrics_k_%s.png" % (str(max_k)))
        fig.savefig(fig_fname)
        
        #save_data(X, 'metrics', metrics_)
        
        for i, labels in enumerate(clustering_):
            if not os.path.exists(os.path.join(directory,str(i+2))):
                os.makedirs(os.path.join(directory,str(i+2)))
            plot_states_matrices(X, 
                                 labels,
                                 save_path=os.path.join(directory,str(i+2)),
                                 save_name_condition=cond)
            
        
        # Save filtered data
        X_fname = os.path.join(directory, "filtered_data.mat")
        save_data(X, 'filtered_data', X_fname)
        
        result[cond] = {'X': X, 'clustering':clustering_, 'data':data_}
        
    return result
    
        
        

def get_results(**kwargs):
    """
    Gets the labels from precomputer clustering
    """
    configuration = {
                     'path':'/media/robbis/DATA/fmri/movie_viviana/',
                     'band':'alpha',
                     'filetype' : 'masked',
                     'fname': 'mat_corr_sub_%s.mat',
                     'conditions' : ['movie', 'scramble', 'rest'],
                     'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
                     'max_k':15,
                     'method':'variance+mean',               
                     
                     'filter':'none' 
                     }
    
    
    configuration.update(kwargs)
    
    path = os.path.join(configuration['path'], configuration['band'])
    filetype = configuration['filetype']
    method = configuration['method']
    max_k = configuration['max_k']
    conditions = configuration['conditions']
    filter_type = configuration['filter']
    
    clustering = dict()
    X = dict()
    
    for condition in conditions:
    
        path_cluster = "%s/%s/%s/%s" % (filetype, method, filter_type, condition)
        path_cluster = os.path.join(path, path_cluster)
        fname = configuration['state_res_fname'] % (condition, 
                                                    str(max_k), 
                                                    method, 
                                                    filetype)
        
        
        path_file = os.path.join(path_cluster, fname)
        clustering_ = pickle.load(file(path_file, 'r'))
        clustering[condition] = clustering_
        
        X[condition] = loadmat(os.path.join(path_cluster, "filtered_data.mat"))['filtered_data']
        
    
    return X, clustering





def get_centers(n_cluster, **kwargs):
    
    conf = {'path':'/media/robbis/DATA/fmri/movie_viviana/',
              'filetype' : 'masked',
              'fname': 'mat_corr_sub_%s.mat',
              'conditions' : ['movie', 'scramble', 'rest'],
              'state_res_fname' : "clustering_labels_%s_maxk_%s_%s_%s.pyobj",
              'max_k':15,
              'method':'speed'                 
            }
    
    
    conditions = conf['conditions']
    conf.update(kwargs)
    
    centers = []
     
    for c in conditions:

        clustering = get_clustering(condition=c, **conf)
        
        data_fname = os.path.join(conf['path'], conf['filetype'], conf['fname'])
        data_, _ = get_data(data_fname % str.upper(c))
        subj_min_speed, _ = get_subsampling_measure(conf['method'])(data_)
        X = data_[subj_min_speed]
        
        centroids = get_centroids(X, clustering[n_cluster-2])
        
        centers.append(centroids)
        
        
    return centers, conditions