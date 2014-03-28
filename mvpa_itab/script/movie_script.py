from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage
from scipy.io import loadmat
import nibabel as ni
from sklearn import linear_model, svm
from sklearn.utils import check_random_state
from sklearn.cross_validation import KFold
from sklearn.feature_selection import f_regression
from scipy.stats.mstats import zscore as sc_zscore
import nibabel

from nilearn import decoding
import nilearn.masking

from mvpa2.suite import *

datapath = '/media/DATA/fmri/movie_viviana/'

filelist = os.listdir(datapath)
filelist = [f for f in filelist if f.find('.mat') != -1]

conditions = ['natvis', 'rest']
bands = ['alpha']

target_list = []
sample_list = []

labels = np.loadtxt(os.path.join(datapath, 'roi_labels.txt'), dtype=np.str_, delimiter='%')
labels = labels.T[1]
subject_list_chunks = np.loadtxt(os.path.join(datapath, 'subj_list'), dtype=np.str)

mask = np.ones(len(labels), np.bool)
#mask[10:] = False
mask_roi = np.meshgrid(mask,mask)[1] * np.meshgrid(mask,mask)[0]

for cond in conditions:
    for band in bands:
        filt_list = [f for f in filelist if f.find(cond) != -1 and f.find(band) != -1]
        data = loadmat(os.path.join(datapath, filt_list[0]))
        
        mat_ = data[data.keys()[0]]
        mat_[np.isinf(mat_)] = 0
        
        masked_mat = mat_ * mask_roi
         
        triu_mat = np.array([np.triu(m) for m in masked_mat])
        samples = np.array([m[np.nonzero(m)] for m in triu_mat])
        targets = [cond for i in samples]
        
        target_list.append(targets)
        sample_list.append(samples)

        
targets = np.hstack(target_list)
samples = np.vstack(sample_list)
chunks = np.hstack((subject_list_chunks, subject_list_chunks))

zsamples = sc_zscore(samples, axis=1)

ds = dataset_wizard(zsamples, targets=targets, chunks=chunks)

zscore(ds)

n_folds = [1,2,3]
#n_feats = np.arange(10, 1220, 50)
n_feats = [10]
err_lst = []

for k in n_folds:
    for n in n_feats:
        #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),
        #                                       FixedNElementTailSelector(
        #                                                               n, mode = 'select',tail = 'upper'))
        
        rfesvm_split = SplitClassifier(LinearCSVMC())
        
        fsel = RFE(rfesvm_split.get_sensitivity_analyzer(
                        postproc=ChainMapper([FxMapper('features', l2_normed),
                                              FxMapper('samples', np.mean),
                                              FxMapper('samples', np.abs)])),
                  ConfusionBasedError(rfesvm_split, confusion_state='stats'),
                  Repeater(2),
                  fselector=FractionTailSelector(0.50,mode='select', tail='upper'),
                  stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),
                  train_pmeasure=False,
                  update_sensitivity=True)
               
        
        clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities','estimates']) 
        
        fclf = FeatureSelectionClassifier(clf, fsel)
        
        
        cvte = CrossValidation(fclf, NFoldPartitioner(cvtype=k, attr='chunks'), enable_ca=['stats'])
        
        err = np.mean(cvte(ds))
        
        err_lst.append([k, n, err])
        
        print('------------------------------------')
        print('n_folds = %d, n_feats = %d' %( k, n))
        print cvte.ca.stats
        
        sensana = fclf.get_sensitivity_analyzer()
        
        weights = sensana(ds)
        
        m = triu_mat[0]
        matrix = np.zeros_like(m)
        
        matrix[m!=0]=(weights.samples[0] - weights.samples[0].mean())/weights.samples[0].std()
        
        thr_matrix = matrix * (np.abs(matrix) > 1.96)
        
        pl.figure(figsize=(13,13))
        pl.imshow(matrix, interpolation='nearest',cmap=pl.cm.seismic)
        pl.xticks(np.arange(len(labels)), labels, rotation='vertical')
        pl.yticks(np.arange(len(labels)), labels)
        pl.colorbar()
        pl.savefig(os.path.join(datapath,'results','movie_weight_matrix_'+str(k)+'_folds_fsel.png'))
        
        pl.figure(figsize=(13,13))
        pl.imshow(thr_matrix, interpolation='nearest',cmap=pl.cm.seismic)
        pl.xticks(np.arange(len(labels)), labels, rotation='vertical')
        pl.yticks(np.arange(len(labels)), labels)
        pl.colorbar()
        pl.savefig(os.path.join(datapath,'results','movie_weight_thresholded_matrix_'+str(k)+'_folds_fsel.png'))

