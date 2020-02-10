import scipy
from scipy.spatial.distance import pdist, squareform, euclidean
from sklearn import metrics
from pyitab.analysis.states.metrics import *
from sklearn.datasets.samples_generator import make_blobs

default_metrics = {'Silhouette': metrics.silhouette_score,
                    'Krzanowski-Lai': kl_criterion,
                    'Global Explained Variance':global_explained_variance,
                    'Within Group Sum of Squares': wgss,
                    'Explained Variance':explained_variance,
                    'Index I': index_i,
                    "Cross-validation":cross_validation_index
                    }

def test_metrics():
    centers = [[-3, 5], [-1, -1], [1, -1], [2,3], [-5, 0], [1,5]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3,
                                random_state=0)
    
    clustering_ = []
    for k in range(2,11):
        
        km = KMeans(n_clusters=k).fit(X)
        labels = km.labels_
        clustering_.append(labels)
        
    
    metrics_ = []
    
    for i, label in enumerate(clustering_):
        metric_values = {}
        for name, metric in default_metrics.items():
            if name == 'Krzanowski-Lai':
                if i == 0 or i == len(clustering_) - 1:
                    prev_labels = None
                    next_labels = None
                else:
                    prev_labels = clustering_[i-1]
                    next_labels = clustering_[i+1]
                
                m = metric(X, 
                           label, 
                           previous_labels=prev_labels, 
                           next_labels=next_labels,
                           precomputed=False)
            else:
                m = metric(X, label)
            
            metric_values[name] = m
        metrics_.append((metric_values.copy()))


        

        
    