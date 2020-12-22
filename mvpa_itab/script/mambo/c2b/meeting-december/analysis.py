###################
from sklearn.model_selection import *

from sklearn.svm.classes import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


import numpy as np
from pyitab.io.loader import DataLoader
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, SampleSlicer, \
    TargetTransformer, Transformer
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
 
conf_file = "/media/robbis/DATA/meg/c2b/meeting-december-data/bids.conf"

loader = DataLoader(configuration_file=conf_file, 
                    loader='bids-meg',
                    bids_window='300',
                    bids_ses='01',
                    task='power')

ds = loader.fetch(subject_names=['sub-109123'], prepro=[Transformer()])
    
_default_options = {
                       
                       'loader__bids_ses': ['01', '02'],
                       
                       'sample_slicer__targets' : [
                           ['LH', 'RH'], 
                           ['LF', 'RF'], 
                           #['LH', 'RH', 'LF', 'RF']
                        ],

                       'estimator__clf': [
                           LinearModel(LogisticRegression(C=1, solver='liblinear')),
                           SVC(C=1, kernel='linear', probability=True),
                           SVC(C=1, gamma=1, kernel='rbf', probability=True),
                           GaussianProcessClassifier(1 * RBF(1.))
                        ],                          
                    }    
    
_default_config = {
               
                       
                       'loader': DataLoader, 
                       'loader__configuration_file': conf_file, 
                       'loader__loader': 'bids-meg', 
                       'loader__bids_window': '300',
                       'loader__task': 'power',
                       'loader__load_fx': 'hcp-motor'
                       'fetch__subject_names': ['sub-109123'],
                       'fetch__prepro': [Transformer()],
                     
                       
                       'prepro':['sample_slicer', 'balancer'],
                       'balancer__attr':'all',

                       'estimator': [('fsel', SelectKBest(k=50)),
                                     ('clf', SVC(C=1, kernel='linear'))],

                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,
                       #'cv__test_size': 0.25,

                       'scores' : ['accuracy'],

                       'analysis': TemporalDecoding,
                       'analysis__n_jobs': 8,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       #'kwargs__cv_attr':'subjects',

                    }
 
estimators = []
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="cross+session").fit(ds=None, **kwargs)
    a.save(save_estimator=True)
    est = a._estimator.scores['mask-matrix_values_value-1.0'][0]['estimator']
    estimators.append(est)
    del a


#########################
# Prediction timecourse #

# Plot results

# See results.py
colormap = {'LH':'navy', 'RH':'firebrick', 'LF':'cornflowerblue', 'RF':'salmon'}
colors = [colormap[t] for t in y]

# Load second session
loader = DataLoader(configuration_file=conf_file, 
                    loader='bids-meg',
                    bids_window='300',
                    bids_ses='02',
                    task='power')

ds_session = loader.fetch(subject_names=['sub-109123'], prepro=[Transformer()])
ds_session = SampleSlicer(targets=['LH', 'RH', 'LF', 'RF']).transform(ds_session)



# 1. How well can we predict the second session giving classifier trained 
#    on the first session?
from sklearn.preprocessing import LabelEncoder

# Estimators loading
from joblib import load
import glob
folders = glob.glob("/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/pipeline-meeting*")
folder.sort()
subject = '109123'
estimators = []
for f in folders:
    folder = os.path.join(f, subject)
    fname = glob.glob(folder+'/*pickle')[0]
    targets = get_dictionary(fname)['targets'].split("+")
    estimator = load(fname)
    estimators.append([targets, estimator['estimator']])


session_generalization_accuracy = []
for targets, estimator in estimators:

    ds_ = SampleSlicer(targets=targets).transform(ds_session)
    X = ds_.samples.copy()
    y = ds_.targets.copy()

    y_ = LabelEncoder().fit_transform(y)

    accuracy = np.mean([est.score(X, y_) for est in estimator], axis=0)

    session_generalization_accuracy.append(accuracy)


for t, targets in enumerate(conditions):

    for i in range(4):
        idx_estimator = i * len(conditions) + t
        estimator = estimators[idx_estimator]
        print(targets, estimator[0].base_estimator['clf'])




## Plot
xticklabels = np.array(["{:5.2f}".format(i) for i in ds.a.times])
yticklabels = np.array(["{:5.2f}".format(i) for i in ds.a.times])  

session_generalization_accuracy = load("/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/session-generalization-accuracy.pickle")
#dump(session_generalization_accuracy, 
#    "/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/session-generalization-accuracy.pickle")
for i, (targets, estimator) in enumerate(estimators):
    
    class_ = "-".join(targets)
    clf = str(estimator[0].base_estimator['clf'])
    kernel = ''
    if clf.find('SVC') != -1 and clf.find("kernel") != -1:
        kernel = clf[clf.find("kernel='"):clf.find(", p")]
    clf = clf[:clf.find('(')] + " " + kernel

    print(clf, class_, i%4)
    
    if i%4 == 0:
        fig, axes = pl.subplots(2, 4, sharex=True)

    r = i%4

    limits = (0.45, .99)
    if i/4 == 3:
        limits = (.25, .6)

    accuracy = session_generalization_accuracy[i]
    im = axes[0, r].imshow(accuracy, 
                          origin='lower',
                          cmap=pl.cm.magma,
                          vmin=limits[0],
                          vmax=limits[1]
                          )

    axes[0, r].set_title("%s | %s" % (clf, class_))
    axes[0, r].set_xticks(np.arange(58)[::8])
    axes[0, r].set_xticklabels(xticklabels[::8])
    axes[0, r].set_xlabel('Training time')

    axes[0, r].set_yticks(np.arange(58)[::8])
    axes[0, r].set_yticklabels(xticklabels[::8])
    axes[0, r].set_ylabel('Testing time')
    axes[0, r].vlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')
    axes[0, r].hlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')

    axes[1, r].plot(np.diag(accuracy))
    axes[1, r].set_xlabel('Training time')
    axes[1, r].set_ylabel('Classification accuracy')
    axes[1, r].set_ylim(limits)
    axes[1, r].vlines(7.5, limits[0], limits[1], colors='r', linestyles='dashed')
    
    if i/4 == 2:
        fig.colorbar(im, ax=axes[:], location='right')

# 2. How well can the classifier generalize using the whole dataset?



# 3. How well can I understand the brain state as function of distance?

# Estimators loading
from joblib import load
import glob
folders = glob.glob("/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/pipeline-*")
subject = '109123'
estimators = []
for f in folders:
    folder = os.path.join(f, subject)
    fname = glob.glob(folder+'/*pickle')[0]
    targets = get_dictionary(fname)['targets'].split("+")
    session = get_dictionary(fname)['ses'].split("+")
    estimator = load(fname)
    estimators.append([targets, estimator['estimator']])


X = ds_session.samples.copy()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.metrics import *

predictions = []
for targets, estimator in estimators[:-3]:
    le.fit(targets)
    
    y_targets = ds_session.targets.copy()
    mask = [y in targets for y in y_targets]
    y_targets[mask] = le.transform(y_targets[mask])
    y_targets[np.logical_not(mask)] = -1

    y_targets = np.int_(y_targets)
    
    try:
        prediction = [est.decision_function(X) for est in estimator]
    except Exception as exc:
        prediction = [est.predict_proba(X) for est in estimator]

    predictions.append([prediction, 
                        y_targets.copy(), 
                        "+".join(targets), 
                        str(estimator[0].base_estimator['clf'])])
    
    print(estimator, targets, np.shape(prediction))

#dump(session_generalization_accuracy, 
#    "/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/predictions.pickle")

# 1. range probabilities
probabilities = np.linspace(0.3, 0.99, 10)
# 2. range distances
distances = np.linspace(0.2, 2.5, 10)

classifier_results = []
for row in predictions:
    
    prediction, y_true, conditions, estimator = row
    print(conditions, estimator, np.shape(prediction))
    # prediction.shape => folds x trials x timepoint x timepoint (x classes)

    idx = np.diag_indices(58)
    prediction = np.array(prediction)
    if len(prediction.shape) == 5:
        prediction = prediction[:, :, idx[0], idx[1], :]
    else:
        prediction = prediction[:, :, idx[0], idx[1]]

    if estimator.find("SVC") != -1 or estimator.find("Linear") != -1:
        # prediction => folds x trials x timepoints -- diagonal classifier
        measures = distances
        compare = compare_distances
    else:
        # predictions => folds x trials x timepoints x classes -- diagonal classifier       
        measures = probabilities
        compare = compare_probabilities

    
    for threshold in measures:
        y_transformed = [compare(p, threshold) for p in prediction]
        # y_transformed => folds x trials x timepoints -- quasi binary

        accuracies = [scorer(y_pred, y_true, fx=confusion_matrix) for y_pred in y_transformed]
        # accuracies => folds x timepoints

        accuracy = np.mean(accuracies, axis=0)

        classifier_results.append([conditions, estimator, threshold, accuracy])



def compare_distances(y_pred, threshold):

    if len(y_pred.shape) == 3:
         # trials x timepoints
        p_pred = np.max(np.abs(y_pred), axis=2)
        y_pred = np.argmax(np.abs(y_pred), axis=2)
        
        y_transformed = np.zeros_like(y_pred)
        far_prediction_mask = np.abs(p_pred) > threshold
        y_transformed[far_prediction_mask] = y_pred[far_prediction_mask]

    else: 

        y_transformed = np.zeros_like(y_pred)

        far_prediction_mask = np.abs(y_pred) > threshold
        y_transformed[far_prediction_mask] = np.sign(y_pred[far_prediction_mask])
        y_transformed[y_transformed == -1] = 0

    y_transformed[np.logical_not(far_prediction_mask)] = -1

    return y_transformed



def compare_probabilities(y_pred, threshold):
    # y_pred = trials x timepoints x classes (probabilities)
    y_ = np.argmax(y_pred, axis=2) # trials x timepoints
    p_pred = np.max(y_pred, axis=2)

    y_transformed = np.zeros_like(y_)

    likely_predictions_mask = p_pred > threshold
    y_transformed[likely_predictions_mask] = y_[likely_predictions_mask]
    y_transformed[np.logical_not(likely_predictions_mask)] = -1 

    return y_transformed



def scorer(y_pred, y_true, metric=accuracy_score):
    return np.array([metric(y, y_true) for y in y_pred.T])


dataframe = pd.DataFrame(classifier_results, columns=['conditions', 'classifier', 'measure', 'accuracies'])


times = np.hstack([timevec for _ in range(15)])


fig, axes = pl.subplots(3, 4)
for i in range(12):
    df = dataframe[i*15:(i+1)*15]
    df = df.explode('accuracies')
    df['accuracies'] = np.float_(df['accuracies'])
    df['time'] = times
    df_pivot = df.pivot('measure', 'time', 'accuracies')
    
    conditions = df['conditions'].values[0].replace("+", '-')
    clf = df['classifier'].values[0]
    kernel = ''
    if clf.find('SVC') != -1 and clf.find("kernel") != -1:
        kernel = clf[clf.find("kernel='"):clf.find(", p")]
    clf = clf[:clf.find('(')] + " " + kernel

    print(conditions, clf)
    
    xticklabels = np.array(["{:5.2f}".format(i) for i in df_pivot.columns])
    yticklabels = np.array(["{:5.2f}".format(i) for i in df_pivot.index])  

    matrix = df_pivot.values

    ax = axes[np.int(np.floor(i/4)), i%4]

    ims = ax.imshow(matrix, origin='lower', aspect='auto', cmap=pl.cm.viridis)
    ax.set_xticks(np.arange(58)[::8])
    ax.set_xticklabels(xticklabels[::8])

    ax.set_yticks(np.arange(15)[::4])
    ax.set_yticklabels(yticklabels[::4])

    ax.vlines(7.5, np.min(df_pivot.index), np.max(df_pivot.index), colors='r', linestyles='dashed')

    ax.set_title(' %s | %s ' %(conditions, clf))

    fig.colorbar(imsh, ax=ax)



    sns.heatmap(df_pivot, 
                xticklabels=xticks_labels, 
                yticklabels=yticks_labels,
                #ax=axes[i%4, np.floor(i/4)]
                )


    




