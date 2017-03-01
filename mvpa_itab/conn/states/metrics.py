from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform, euclidean
import logging

logger = logging.getLogger(__name__)

def ch_criterion(X, labels):
    
    k = len(np.unique(labels))
    n = X.shape[0]
    
    b = bgss(X, labels)
    w = wgss(X, labels)
    
    return (n - k)/(k - 1) * b / w


def bgss(X, labels):
    
    ds_mean = X.mean(0)
    ss = 0
    for i in np.unique(labels):
        cluster_data = X[labels == i]
        ss += (euclidean(cluster_data.mean(0), ds_mean) ** 2) * cluster_data.shape[0]
        
    return ss



def wgss(X, labels):
    
    ss = 0
    for i in np.unique(labels):
        cluster_data = X[labels == i]
        cluster_mean = cluster_data.mean(0)
        css = 0
        for x in cluster_data:
            css += euclidean(x, cluster_mean) ** 2
        
        ss += css
        
    return ss
            


def kl_criterion(X, labels, previous_labels=None, next_labels=None, precomputed=True):
    
    n_cluster = len(np.unique(labels))
    
    """
    if n_cluster <= 1 or previous_labels==None or next_labels==None:
        return 0
    """
    
    n_prev_clusters = len(np.unique(previous_labels))
    n_next_clusters = len(np.unique(next_labels))
    
    if n_cluster != n_next_clusters-1 or n_prev_clusters+1 != n_cluster:
        return 0
    
    M_previous = m(X, previous_labels, precomputed=precomputed)
    M_next = m(X, next_labels, precomputed=precomputed)
    M_current = m(X, labels, precomputed=precomputed)

    return 1 - 2 * M_current/M_previous + M_next/M_previous



def W(X, labels, precomputed=True):

    #distance = squareform(pdist(X, 'euclidean'))
    import itertools
    w_ = 0
    for k in np.unique(labels):
        cluster_ = labels == k
        
        if precomputed == True:
            index_cluster = np.nonzero(cluster_)
            combinations_ = itertools.combinations(index_cluster[0], 2)
            nrow = cluster_.shape[0]
            array_indices = [get_triu_array_index(n[0], n[1], nrow) for n in combinations_]
            cluster_dispersion = X[array_indices].sum()
        else:
            
            X_k = X[cluster_]
            cluster_distance = squareform(pdist(X_k, 'euclidean'))
            #cluster_distance = distance[cluster_,:][:,cluster_]
            upper_index = np.triu_indices(cluster_distance.shape[0], k=1)
            cluster_dispersion = cluster_distance[upper_index].sum()
        
        w_ += (cluster_dispersion ** 2) * 0.5 * 1./cluster_.shape[0]
    return w_
    


def m(X, labels, precomputed=True):
    n_cluster = len(np.unique(labels))
    return W(X, labels, precomputed=precomputed) * np.power(n_cluster, 2./X.shape[1])

    
    
def gap(X, labels, nrefs=20, refs=None):
    """
    Compute the Gap statistic for an nxm dataset in X.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of X.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = X.shape
    if refs==None:
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        
    
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
    
    k = len(np.unique(labels))
    kml = labels
    kmc = np.array([X[labels == l].mean(0) for l in np.unique(labels)])

    disp = sum([euclidean(X[m,:], kmc[kml[m],:]) for m in range(shape[0])])

    refdisps = scipy.zeros((rands.shape[2],))
    for j in range(rands.shape[2]):
        km = KMeans(n_clusters=k).fit(rands[:,:,j])
        kml = km.labels_
        kmc = km.cluster_centers_
        refdisps[j] = sum([euclidean(rands[m,:,j], kmc[kml[m],:]) for m in range(shape[0])])
        
    gaps = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
        
    return gaps



def explained_variance(X, labels):
    
    explained_variance_ = 0
    k_ = np.unique(labels).shape[0]
    great_avg = X.mean()
    
    for i in np.unique(labels):
        
        cluster_mask = labels == i
        group_avg = X[cluster_mask].mean()
        n_group = np.count_nonzero(cluster_mask)
        
        group_var = n_group * np.power((group_avg - great_avg), 2) / (k_ - 1)
        
        explained_variance_ += group_var
        
    return explained_variance_
        


def global_explained_variance(X, labels):
    
    centroids = np.array([X[labels == l].mean(0) for l in np.unique(labels)])
    
    global_conn_power = X.std(axis=1)
    
    denominator_ = np.sum(global_conn_power**2)
    
    numerator_ = 0
    
    for i, conn_pwr in enumerate(global_conn_power):
        
        k_map = centroids[labels[i]]
        corr_ = scipy.stats.pearsonr(X[i], k_map)[0]
        
        numerator_ += np.power((conn_pwr * corr_), 2)
        
    return numerator_/denominator_



def get_triu_array_index(i, j, n_row):
    return (n_row*i+j)-np.sum([(s+1) for s in range(i+1)])




def calculate_metrics(X, 
                      clustering_labels, 
                      metrics_kwargs=None):
    
    default_metrics = {'Silhouette': metrics.silhouette_score,
                       'Calinski-Harabasz': ch_criterion, 
                       'Krzanowski-Lai': kl_criterion,
                       'Global Explained Variance':global_explained_variance,
                       #'Gap': gap,
                        }
    
    if metrics_kwargs != None:
        default_metrics.update(metrics_kwargs)
    
    metrics_ = []
    k_step = np.zeros(len(clustering_labels), dtype=np.int8)
    
    for i, label in enumerate(clustering_labels):
        
        k = len(np.unique(label))
        k_step[i] = k
        
        logger.info('Calculating metrics for k: %s' %(str(k)))
        
        metric_list = []
        
        for metric_name, metric_function in default_metrics.items():
            
            logger.info(" - Calculating %s" %(metric_name))
            
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


