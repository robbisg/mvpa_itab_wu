import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from pyitab.ext.sklearn._validation import cross_validate
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

import h5py
import hdf5storage
import numpy as np
from scipy.io import loadmat, savemat

clf = make_pipeline(Normalizer(),  # z-score normalization
                    SelectKBest(f_classif, k=50),  # select features for speed
                    LinearModel(LogisticRegression(C=1, solver='liblinear')))
#time_decod = SlidingEstimator(clf, scoring='accuracy')
time_decod = GeneralizingEstimator(clf, scoring='accuracy')


shared = "/run/user/1000/gvfs/smb-share:server=192.168.30.54,share=meg_data_analisi/HCP_Motor_Task_analysis/109123/"


scores_ses = []
decoders = []

bigdata = []
for f in os.listdir(shared):
    fname = os.path.join(shared, f)
    mat = h5py.File(fname)
    data = mat['powerbox'][:]
    data /= np.nanmean(data)
    data = np.float32(data.swapaxes(1, 2))
    labels = mat['trialvec'][:][0]

    rt = mat['trailinfo'][:][5]

    
    mask = np.logical_and(np.logical_or(labels == 1, 
                                        labels == 4),
                          rt <= 0.5)
    

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

    scores = cross_validate(time_decod, X, y, cv=5, n_jobs=-1, return_estimator=True)

    scores_ses.append(scores)


# Plot accuracy generalization
mat = h5py.File(fname)
t = mat['timevec'][:].T[0]
idx = [0, 50, 100, 150, 200]
idx = [0, 20, 40, 60, 80, 100]
tt = t >= 0
ticklabels = np.array(["{:5.2f}".format(i) for i in t])[tt]


fig, axes = pl.subplots(2, 2)
for i, score in enumerate(scores_ses):

    
    
    accuracy = score['test_score'].mean(0)[:,tt][tt,:]

    ax1 = axes[0, i]
    p = ax1.imshow(accuracy, origin='lower', cmap=pl.cm.magma, vmin=0.4, vmax=1)
    ax1.set_title("Session %d" % (i+1))
    ax1.set_xticks(idx)
    ax1.set_xticklabels(ticklabels[idx])
    ax1.set_xlabel('Training time')

    ax1.set_yticks(idx)
    ax1.set_yticklabels(ticklabels[idx])
    ax1.set_ylabel('Testing time')
    
    #fig.colorbar(p)   

    ax2 = axes[1, i]
    l = ax2.plot(t[tt], np.diag(accuracy))
    ax2.set_ylim((0.37, 0.95))
    ax2.hlines(0.5, -1., 1., linestyle='dashed', color='gray')
    ax2.vlines(0, 0.37, 0.95)
    ax2.set_xlabel('Training time')

    if i == 0:
        ax2.set_ylabel("Accuracy")


fig.colorbar(p)


sessions_feat = []
for i, score in enumerate(scores_ses):
    cv_features = []
    cv_estimators = score['estimator']
    for est in cv_estimators:
        t_features = [e.steps[1][1].get_support() for e in est.estimators_]
        cv_features.append(t_features)

    sessions_feat.append(cv_features)


###################### Session generalization ###########################
bigdata = []
for f in os.listdir(shared):
    fname = os.path.join(shared, f)
    mat = h5py.File(fname)
    data = mat['powerbox'][:]
    data /= np.nanmean(data)
    data = np.float32(data.swapaxes(1, 2))
    labels = mat['trialvec'][:][0]

    rt = mat['trailinfo'][:][5]

    """
    mask = np.logical_and(np.logical_or(labels == 1, 
                                        labels == 4),
                          rt <= 0.5)
    """

    mask = labels != 6

    X = data[mask]
    y = labels[mask]

    bigdata.append([X, y])


# Min samples
ys = [list(Counter(y[1]).values()) for y in bigdata]
nsample = np.min(ys)
resampler_dict = {k: nsample for k in np.unique(bigdata[0][1])}

bigres = []
rus = RandomUnderSampler(random_state=42, sampling_strategy=resampler_dict)
for data in bigdata:
    X, y = data
    _, _ = rus.fit_resample(X[...,0], y)
    X_ = X[rus.sample_indices_]
    y_ = y[rus.sample_indices_]
    bigres.append([X_, y_])

mask = np.logical_or(y_ == 1, y_ == 4)
X = np.dstack((bigres[0][0], bigres[1][0]))[mask]
y = y_[mask]

scores = cross_validate(time_decod, X, y, cv=5, 
                        n_jobs=-1, return_predictions=True, 
                        return_estimator=True)

pl.imshow(scores['test_score'].mean(0), origin='lower')
