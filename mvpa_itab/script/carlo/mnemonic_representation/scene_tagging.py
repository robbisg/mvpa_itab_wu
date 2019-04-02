import pandas as pd
import seaborn as sns
from sklearn.cluster import *

dataframe = pd.read_excel("/home/robbis/Downloads/Boundaries_4Rob.xlsx")

X = dataframe['onset (s)'].values.reshape(-1, 1)

centroids = []
fig = pl.figure(figsize=(13,10))
pl.plot(X, [-1.]*len(X), '|', color='k', ms=15)

#ax = sns.rugplot(X, 0.5, c='k')

from joblib import Parallel, delayed


def idx(idx_1, idx_2):
    import itertools
    for i, j in itertools.product(idx_1, idx_2):
        yield i, j


def clustering(eps, min_samples=9):
    db = DBSCAN(eps=eps+1, min_samples=min_samples+1).fit(X)
    centroids = []
    for l in np.unique(db.labels_)[1:]:
        X_ = X[db.labels_ == l]
        centroids.append(X_.mean())
    
    return centroids



#centroids = Parallel(n_jobs=-1)(delayed(clustering)(i, j) for i, j in idx(20, 30))

centroids = Parallel(n_jobs=-1)(delayed(clustering)(i) for i in np.arange(4, 8))

for i, centroid in enumerate(centroids):
        pl.plot(centroid, [i]*len(centroid), '|', ms=10)
        #sns.rugplot(centroid, offsets=(0, 0.05+i), ax=ax )




dfs = []
for c in centroids:
    dfs.append(pd.DataFrame(c))

times_df = pd.concat([pd.DataFrame(c) for c in centroids],
                    ignore_index=True, axis=1)

times_df.to_excel("/home/robbis/centroids.xlsx")