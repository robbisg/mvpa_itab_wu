import itertools
import numpy as np
import scipy as sp
from scipy import signal
from pyitab.simulation.autoregressive import *
from pyitab.simulation.connectivity import *
from pyitab.analysis.states.metrics import *
from pyitab.analysis.states.subsamplers import *
from pyitab.analysis.states.base import *
from sklearn import cluster, mixture
import scipy

n_nodes = 10
n_brain_states = 6
max_edges = 5
lifetime_range = [2.5, 3.5] # seconds
n_iteration = 1
sample_frequency = 128

import itertools

snr_ratios = [5, 10, 50, 100]
min_lifetimes = [.5, 1, 1.5, 2]
trials = range(16)
p = itertools.product(trials, snr_ratios, min_lifetimes)

from joblib import Parallel, delayed


results = Parallel(n_jobs=10, verbose=1)(delayed(pipeline)(i, snr, min_) for i, (_, snr, min_) in enumerate(p))


def pipeline(i, snr, min_lifetime):
    items = {}
    connectivity_model = ConnectivityStateSimulator(n_nodes=n_nodes,
                                                    max_edges=max_edges,
                                                    fsamp=sample_frequency
                                                    )
                                                    
    _ = connectivity_model.fit(n_brain_states=6, 
                               length_states=10, 
                               min_time=min_lifetime, 
                               max_time=3.5)


    model = PhaseDelayedModel(snr=snr, delay=np.pi*0.5)
    data = model.fit(connectivity_model)

    n_edges = connectivity_model._n_edges

    # Butter filter
    items.update({'bs_length':   connectivity_model._state_length, 
                  'bs_sequence': connectivity_model._state_sequence,
                  'bs_matrices': connectivity_model._states,
                  'bs_dynamics': connectivity_model._dynamics,
                  'signals':     data,
                  })
                      
    
    ###
    from pyitab.preprocessing.filter import ButterFilter
    bfilter = ButterFilter(order=8, min_freq=6, max_freq=20, btype='bandpass')
    data = bfilter.transform(data)


    from pyitab.preprocessing.connectivity import SlidingWindowConnectivity
    conn = SlidingWindowConnectivity(window_length=1)
    data = conn.transform(data)

    cfilter = ButterFilter(order=4, max_freq=2, btype='lowpass')
    data = cfilter.transform(data)

    data.save('./subj_ds-%03d.gzip' % (i), compression='gzip')



metrics_, k_step = calculate_metrics(X, clustering_)

metric_guesses = dict()
for i, (name, values) in enumerate(metrics_.items()):
    if name in ['Silhouette', 'Krzanowski-Lai', 'Index I']:
        guessed_cluster = np.nonzero(np.max(values) == values)[0][0] + k_step[0]
    else:
        data = np.vstack((k_step, values)).T
        theta = np.arctan2(values[-1] - values[0], k_step[-1] - k_step[0])
        co = np.cos(theta)
        si = np.sin(theta)
        rotation_matrix = np.array(((co, -si), (si, co)))
        # rotate data vector
        data = data.dot(rotation_matrix)
        fx = np.max
        if name != 'Global Explained Variance':
            fx = np.min
        guessed_cluster = np.nonzero(data[:, 1] == fx(data[:, 1]))[0][0] + k_step[0]

    metric_guesses[name] = guessed_cluster


results = { 
    'clustering_labels': clustering_,
    'guesses': metric_guesses,
    'metrics': metrics_,
    'connectivity': iplv_filtered,
}

    return items, results



def pipeline(half=1, n_jobs=1):
    # Last one then I will refactor!!

    from sklearn import cluster, mixture


    clustering_algorithms = (
        ('MiniBatchKMeans', cluster.MiniBatchKMeans),
        ('KMeans', cluster.KMeans), 
        ('SpectralClustering', cluster.SpectralClustering),
        ('Ward', cluster.AgglomerativeClustering),
        ('Birch', cluster.Birch),
        #('GaussianMixture', mixture.GaussianMixture)
    )
    

    def _parallel(i, mat):

        connectivity = mat['connectivity'][0][0]
        connectivity = np.expand_dims(connectivity, axis=0)
        results = []
        for name, method in clustering_algorithms:
            item = {}

            X, full_clustering, clustering_, centroids = \
                cluster_state(connectivity, k_range=range(2, 10), method=method)
        
            metrics_, k_step = calculate_metrics(X, clustering_)

            item['algorithm'] = name
            item['clustering'] = full_clustering
            item['centroids'] = centroids

            metric_guesses = dict()
        
            for i, (name, values) in enumerate(metrics_.items()):
                if name in ['Silhouette', 'Krzanowski-Lai', 'Index I']:
                    guessed_cluster = np.nonzero(np.max(values) == values)[0][0] + k_step[0]
                else:
                    data = np.vstack((k_step, values)).T
                    theta = np.arctan2(values[-1] - values[0], k_step[-1] - k_step[0])
                    co = np.cos(theta)
                    si = np.sin(theta)
                    rotation_matrix = np.array(((co, -si), (si, co)))
                    # rotate data vector
                    data = data.dot(rotation_matrix)
                    fx = np.max
                    if name != 'Global Explained Variance':
                        fx = np.min
                    guessed_cluster = np.nonzero(data[:, 1] == fx(data[:, 1]))[0][0] + k_step[0]

                metric_guesses[name] = guessed_cluster
            
            item['guess'] = metric_guesses
            results.append(item.copy())

        return results

    data = scipy.io.loadmat("/m/home/home9/97/guidotr1/unix/results/c2b/data/clustering200_%s.mat" % (str(half)))

    data = data['clustering'][0]
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel)(i, mat) for i, mat in enumerate(data))

    return results


def cluster_state(data, k_range=range(2, 15), method=cluster.KMeans):
    """
    This performs either preprocessing and clustering
    using a range of k-clusters.
    
    Returns the preprocessed dataset and a list of labels.
    """

    subsampler = VarianceSubsampler()
    X = subsampler.fit_transform(data)
    clustering = []
    full_clustering = []
    centroids = []
    
    if k_range[0] < 2:
        k_range = range(2, k_range[-1])
    
    for k in k_range:
        #print(method)
        km = method(n_clusters=k).fit(X)
        clustering.append(km.labels_)
        try:
            labels_ = km.predict(np.squeeze(data))
        except AttributeError as err:
            if hasattr(km, 'cluster_centers_'):
                centers = km.cluster_centers_
            else:
                centers = np.array([X[km.labels_ == l].mean(0) for l in np.unique(km.labels_)])
            C = np.squeeze(data)
            D = np.dot(C, centers.T)
            y, x = np.where(D.T == D.max(1))

            labels_ = np.zeros(D.shape[0])
            labels_[x] = y
        
        centers = np.array([X[km.labels_ == l].mean(0) for l in np.unique(km.labels_)])

        centroids.append(centers)
        full_clustering.append(labels_)
        
    return X, full_clustering, clustering, centroids


import pandas as pd

item = dict()
for j, repetition in enumerate(results+results2):
    for alg in repetition:
        method = alg['algorithm']
        if j == 0:
            item[method] = dict()
        
        for name, guess in alg['guess'].items():
            n = np.int(guess == 6)

            if j == 0:
                item[method][name] = 0
            
            item[method][name] += n

df = pd.DataFrame(item)
df = df.sort_values(by=['SpectralClustering'], ascending=False)

simulated_data = scipy.io.loadmat('/m/home/home9/97/guidotr1/unix/results/c2b/data/data200.mat')
simulated_data = simulated_data['simulated_data'][0]


errors = np.zeros((200, 5, 2))


Parallel(n_jobs=10, verbose=1)(delayed(_error_parallel)(i, sim_data, errors) for i, sim_data in enumerate(simulated_data))



def _error_parallel(i, sim_data, errors):
    
    clu_data = results[i]
    true_dynamics = sim_data['bs_dynamics'][0][0][0]
    true_centers = sim_data['bs_matrices'][0][0]
    
    for j, res in enumerate(clu_data):

        derror = get_error_dynamics(res['clustering'][4], true_dynamics)
        merror = get_error_matrices(res['centroids'][4], true_centers)

        errors[i, j, 0] = derror
        errors[i, j, 1] = merror


def get_error_dynamics(clustering_dynamics, true_dynamics):
    
    cluster_idx = np.unique(true_dynamics)
    
    assert len(clustering_dynamics) == len(true_dynamics) - 255

    cluster_binary = np.zeros((len(cluster_idx), len(clustering_dynamics)), dtype=np.bool)
    for i in cluster_idx: 
        cluster_binary[i] = clustering_dynamics == i

    permuted_dynamics = np.zeros_like(clustering_dynamics)
    min_error = 1e6
    perm = itertools.permutations(cluster_idx)
    for p in perm:
        x, y = np.nonzero(cluster_binary[p, :])
        permuted_dynamics[y] = x

        error = np.sum(np.abs(permuted_dynamics == true_dynamics[:-255]))

        if error < min_error:
            min_error = error
            min_p = p

    return min_error / len(clustering_dynamics)


def get_error_matrices(clustering_centers, true_centers):
    # Center or vector? use the same name for same shapes!

    triu = np.triu_indices(true_centers.shape[1], k=1)
    true_vector = np.array([m[triu] for m in true_centers])

    similarity = np.dot(true_vector, clustering_centers.T)
    t_similarity = np.diag(np.sqrt(np.dot(true_vector, 
                                          true_vector.T)))
    c_similarity = np.diag(np.sqrt(np.dot(clustering_centers, 
                                          clustering_centers.T)))

    
    idx_true, idx_clustering = np.nonzero(similarity == similarity.max(0))
    norm_similarity = []
    for x, y in zip(idx_true, idx_clustering):
        n = similarity[x, y]/(t_similarity[x] * c_similarity[y])
        norm_similarity.append(n)

    norm_similarity = np.array(norm_similarity).mean()

    return norm_similarity