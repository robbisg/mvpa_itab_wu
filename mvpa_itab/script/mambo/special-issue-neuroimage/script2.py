from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from pyitab.ext.sklearn._validation import cross_validate
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

import h5py
import hdf5storage
import numpy as np
from scipy.io import loadmat, savemat

clf = make_pipeline(Normalizer(),  # z-score normalization
                    SelectKBest(f_classif, k=50),  # select features for speed
                    LinearModel(LogisticRegression(C=1, solver='liblinear')))
#time_decod = SlidingEstimator(clf, scoring='accuracy')
time_decod = GeneralizingEstimator(clf, scoring='accuracy')


shared = "/media/robbis/DATA/fmri/neuroimage-special-issue/"
files = os.listdir(shared)
files = [f for f in files if f.find('.mat') != -1]
scores_ses = []
decoders = []

bigdata = []
for f in files:
    fname = os.path.join(shared, f)
    mat = h5py.File(fname)
    data = mat['powerbox'][:]
    data /= np.nanmean(data)
    data = np.float32(data.swapaxes(1, 2))
    labels = mat['trialvec'][:][0]

    rt = mat['trailinfo'][:][5]

    
    mask = np.logical_or(labels == 1, labels == 4)
    

    #mask = labels != 6

    X = data[mask]
    y = labels[mask]

    print(Counter(y))

    rus = RandomUnderSampler(random_state=42)
    _, y = rus.fit_sample(X[..., 0], y)

    X = X[rus.sample_indices_]

    print(Counter(y))

    del data
    del mat

    scores = cross_validate(time_decod, X[...,:-3], y, cv=5, n_jobs=-1, return_estimator=True)

    scores_ses.append((f, scores))


############################
windows = [.2, .3, .4, .5]
for i, (f, s) in enumerate(scores_ses):
    score = s['test_score']
    fname = os.path.join(shared, f)
    mat = h5py.File(fname)
    times = np.squeeze(mat['timevec'].value)

    if i == 3:
        times = times[:-3]

    avg = np.diag(score.mean(0))
    pl.plot(times + .5*windows[i], avg, label=windows[i])

pl.legend()
    


