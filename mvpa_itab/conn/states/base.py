import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema, argrelmin
from scipy.spatial.distance import pdist, squareform
import logging
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


logger = logging.getLogger(__name__)


def clustering(X, n_cluster):
    """
    Function used to cluster data using kmeans and
    a fixed number of cluster.
    
    Returns the labels
    """
    
    km = KMeans(n_clusters=n_cluster).fit(X)
    return km.labels_



def cluster_state(X, k_range=range(2, 15)):
    """
    This performs either preprocessing and clustering
    using a range of k-clusters.
    
    Returns the preprocessed dataset and a list of labels.
    """

    
    clustering_ = []
    
    if k_range[0] < 2:
        k_range = range(2, k_range[-1])
    
    for k in k_range:
        logger.info('Clustering with k: '+str(k))
        labels = clustering(X, k)
        clustering_.append(labels)
        
    return X, clustering_



def get_extrema_histogram(arg_extrema, n_timepoints):
    
    hist_arg = np.zeros(n_timepoints)
    n_subjects = len(np.unique(arg_extrema[0]))
    
    for i in range(n_subjects):
        sub_max_arg = arg_extrema[1][arg_extrema[0] == i]
        hist_arg[sub_max_arg] += 1
        
    return hist_arg



def subsample_data(data, method='speed', peak='min'):
    """
    Function used to select timepoints using 
    speed methods (low velocity states) or 
    variance methods (high variable states)
    
    Returns the preprocessed dataset
    """
    
    peak_mapper = {'max': np.greater_equal,
                   'min': np.less_equal}
    
    
    
    method = get_subsampling_measure(method)
    _, measure = method(data)
    
    peaks = argrelextrema(measure, peak_mapper[peak], axis=1, order=5)
    
    X = data[peaks]
    
    
    return X



def get_subsampling_measure(method):
    
    method_mapping = {
                      'speed': get_min_speed_arguments,
                      'variance': get_max_variance_arguments
                      }
    
    
    return method_mapping[method]



def get_max_variance_arguments(data):
    """
    From the data it extract the points with high local variance 
    and returns the arguments of these points and the 
    variance for each point.
    """
    
    stdev_data = data.std(axis=2)   
    arg_maxima = argrelextrema(np.array(stdev_data), np.greater, axis=1)
    
    
    return arg_maxima, stdev_data
    


def get_min_speed_arguments(data):
    """
    From the data it extract the points with low local velocity 
    and returns the arguments of these points and the 
    speed for each point.    
    """
    
    subj_speed = []
    for i in range(data.shape[0]):
        distance_ = squareform(pdist(data[i], 'euclidean'))
        
        speed_ = [distance_[i, i+1] for i in range(distance_.shape[0]-1)]
        subj_speed.append(np.array(speed_))
    
    subj_speed = np.vstack(subj_speed)
    subj_min_speed = argrelmin(np.array(subj_speed), axis=1)
    
    return subj_min_speed, subj_speed


def gaussian_kernel(dist, dc):
    n_samples = dist.shape[0]
    
    rho = np.zeros(n_samples)
    
    m_indices = np.triu_indices(n_samples, k=1)
    
    # Gaussian kernel
    for i, j in np.vstack(m_indices).T:
        gaussian_kernel = np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
        rho[i] = rho[i] + gaussian_kernel
        rho[j] = rho[j] + gaussian_kernel
    
    
    return rho



def cutoff(dist, dc):
    n_samples = dist.shape[0]
    
    rho = np.zeros(n_samples)
    
    m_indices = np.triu_indices(n_samples, k=1)
    
    # Gaussian kernel
    for i, j in np.vstack(m_indices).T:
        if dist[i,j] < dc:
            rho[i] += 1
            rho[j] += 1
    
    
    return rho    



class PeakDensityClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    
    def __init__(self, dc='percentage', percentage=2., cluster_threshold=12., rhofx=gaussian_kernel):
        
        if dc != 'percentage':
            self.dc = dc
        else:
            self.dc = 0
            
        self.perc = percentage
        self.cluster_threshold = cluster_threshold
        self.labels_ = None
        self.rhofx = rhofx
        
        
        
    def _get_threshold(self):
        
        if self.cluster_threshold < 1.:
            return np.max(self.rho_) * np.max(self.delta_) * self.cluster_threshold
        
        if self.cluster_threshold < 4.:
            
            metric = self.rho_ * self.delta_
            return np.mean(metric) + self.cluster_threshold * np.std(metric)
        
        else:
            
            return self.cluster_threshold
             
    
    
    
    def _compute_distance(self, X):
        
        n_samples = X.shape[0]
        
        xdist = pdist(X) 
        dist = squareform(xdist)
        
        if self.dc == 0:
            position = int(round(n_samples * self.perc/100.))
            self.dc = np.sort(xdist)[position]
            
        
        return dist
    
    
    def _compute_rho(self, dist, dc):
                
        return self.rhofx(dist, dc)
    
    
    
    def _compute_delta(self, n_samples):
        
        
        rho = self.rho_
        dist = self.dist_
        
        ordrho = np.argsort(rho)[::-1]
        delta = np.zeros(n_samples)
        nneigh = np.zeros(n_samples)
        
        delta[ordrho[0]] = -1
        nneigh[ordrho[0]] = 0
    
        for i in range(n_samples):
            
            min_rho_mask = rho >= rho[i]
            
            min_dist = dist[i][min_rho_mask]
            nonzero = np.nonzero(min_dist)
            delta[i] = np.max(delta)
            if np.count_nonzero(min_rho_mask) != 1:
            
                delta[i] = np.min(min_dist[nonzero])
                ind = np.where(dist == delta[i])
                nneigh[i] = ind[0][0]
                if ind[0][0] == i:
                    nneigh[i] = ind[1][0]
        
        return delta, nneigh
    
    
    
    def _assign_cluster(self, n_samples, cluster_idx):
        
        rho = self.rho_
        dist = self.dist_
        
        clustering = np.zeros_like(rho)
        
          
        clustering = np.zeros_like(rho)
        clustering[cluster_idx] = cluster_idx
    
        for idx in range(n_samples):
            if clustering[idx] == 0:
                argmin = np.argmin(dist[idx, cluster_idx])
                clustering[idx] = cluster_idx[argmin]
    
        clustering = np.int_(clustering)
        
        
        return clustering
        
        
    def _compute_halo(self, n_samples, cluster_idx):
        
        rho = self.rho_
        dist = self.dist_
        dc = self.dc
        
        clustering = self.labels_
        
        halo = self.labels_.copy()
        n_cluster = len(cluster_idx) 
        
        if n_cluster > 1:
            bord_rho = np.zeros(n_cluster)
        
        m_indices = np.vstack(np.tril_indices(n_samples, k=1)).T
        
        # Gaussian kernel
        for i, j in m_indices:
            
            if clustering[i] != clustering[j] and dist[i,j] <= dc:
                rho_aver = 0.5*(rho[i]+rho[j])
    
                idc = np.argwhere(cluster_idx == clustering[i])
                if rho_aver > bord_rho[idc]:
                    bord_rho[idc] = rho_aver
                
                jdc = np.argwhere(cluster_idx == clustering[j])
                if rho_aver > bord_rho[jdc]:
                    bord_rho[jdc] = rho_aver
                
        for i in range(n_samples):
            idc = np.argwhere(cluster_idx == clustering[i])
            if rho[i] < bord_rho[idc]:
                halo[i] = 0
        return
    
    
        
    def fit(self, X, y=None):
        
        n_samples = X.shape[0]
    
        self.dist_ = self._compute_distance(X)
        
        self.rho_ = self._compute_rho(self.dist_, self.dc)

        self.delta_, self.nn_ = self._compute_delta(n_samples)
        
        # Get centers
        self.threshold = self._get_threshold()
        cluster_idx = np.nonzero(self.delta_ * self.rho_ > self.threshold)[0]
        self.cluster_centers_ = X[cluster_idx]
        
       
        self.labels_ = self._assign_cluster(n_samples, cluster_idx)
        
        self.halo_ = self._compute_halo(n_samples, cluster_idx)

        
        return self
    
       

