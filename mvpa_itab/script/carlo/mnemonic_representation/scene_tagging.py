import pandas as pd
import seaborn as sns
from sklearn.cluster import *

dataframe = pd.read_excel("/home/robbis/Downloads/Boundaries_4Rob.xlsx")

X = dataframe['onset (s)'].values.reshape(-1, 1)

centroids = []
pl.plot(X, [-1.]*len(X), '|', color='k', ms=15)

from joblib import Parallel, delayed


def idx(idx_1, idx_2):
    import itertools
    for i, j in itertools.product(idx_1, idx_2):
        yield i, j


def clustering(eps, min_samples):
    db = DBSCAN(eps=eps+1, min_samples=min_samples+1).fit(X)
    centroids = []
    for l in np.unique(db.labels_)[1:]:
        X_ = X[db.labels_ == l]
        centroids.append(X_.mean())
    
    return centroids



centroids = Parallel(n_jobs=-1)(delayed(clustering)(i, j) for i, j in idx(20, 30))




