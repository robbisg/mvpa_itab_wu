import scipy
from scipy.spatial.distance import pdist, squareform, euclidean
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

def test_metrics():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    
    clustering_ = []
    for k in range(2,11):
        
        km = KMeans(n_clusters=k).transform(X)
        labels = km.labels_
        clustering_.append(labels)
        
    
    metrics_ = []
    
    for i, label in enumerate(clustering_):
        
        silhouette_ = metrics.silhouette_score(X, label)
        gap_ = gap(X, label)
        ch_ = ch_criterion(X, label)
        
        if i == 0 or i == len(clustering_) - 1:
            prev_labels = None
            next_labels = None
        else:
            prev_labels = clustering_[i-1]
            next_labels = clustering_[i+1]
            
        
        kl_ = kl_criterion(X, 
                           label, 
                           previous_labels=prev_labels, 
                           next_labels=next_labels)
        
        exp_var = explained_variance(X, label)
        g_exp_var = global_explained_variance(X, label)
        
        metrics_.append([silhouette_,
                         gap_,
                         ch_,
                         kl_,
                         exp_var,
                         g_exp_var])
    
    