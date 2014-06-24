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

import os

from mvpa2.suite import RFE, SplitClassifier, CrossValidation, LinearCSVMC
from mvpa2.suite import zscore, dataset_wizard, ConfusionBasedError, FxMapper
from mvpa2.suite import Repeater, FractionTailSelector, BestDetector, \
                        NFoldPartitioner
from mvpa2.suite import NBackHistoryStopCrit, l2_normed, ChainMapper, \
                        FeatureSelectionClassifier

datapath = '/media/DATA/fmri/movie_viviana/corr_raw/RAW_mat_corr/'

filelist = os.listdir(datapath)
filelist = [f for f in filelist if f.find('.mat') != -1]

conditions = ['movie', 'scramble', 'rest']
bands = ['alpha']#,'beta','gamma','delta','theta']

target_list = []
sample_list = []
chunk_list = []
band_list = []

labels = np.loadtxt(os.path.join(datapath, 'roi_labels.txt'),
                    dtype=np.str_,
                    delimiter='\t')

labels = labels.T[1]
subject_list_chunks = np.loadtxt(os.path.join(datapath, 'subj_list'),
                                 dtype=np.str)

mask = np.ones(len(labels), np.bool)
#mask[10:] = False
mask_roi = np.meshgrid(mask, mask)[1] * np.meshgrid(mask, mask)[0]

for cond in conditions:
    for band in bands:
        filt_list = [f for f in filelist if f.find(cond) != -1 \
                                        and f.find(band) != -1]
        data = loadmat(os.path.join(datapath, filt_list[0]))

        mat_ = data[data.keys()[0]]

        #mat_[np.isinf(mat_)] = 0

        il = np.tril_indices(mat_[0].shape[0])

        #masked_mat = mat_ * mask_roi
        masked_mat = mat_ * mask_roi[np.newaxis,:]
        for m in masked_mat:
            m[il] = 0

        #samples = np.array([m[il] = 0 for m in masked_mat])
        samples = np.array([m[np.nonzero(m)] for m in masked_mat])
        targets = [cond for i in samples]

        band_ = [band for i in samples]

        target_list.append(targets)
        sample_list.append(samples)
        chunk_list.append(subject_list_chunks)
        band_list.append(band_)

targets = np.hstack(target_list)
samples = np.vstack(sample_list)
chunks = np.hstack(chunk_list)

zsamples = sc_zscore(samples, axis=0)

ds = dataset_wizard(zsamples, targets=targets, chunks=chunks)
ds.sa['band'] = np.hstack(band_list)

zscore(ds)

n_folds = [4]
#n_feats = np.arange(10, 1220, 50)
n_feats = [10]
err_lst = []

sens_mat = []

for k in n_folds:
    for n in n_feats:
        #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),
        #                                       FixedNElementTailSelector(
        #                                                               n, mode = 'select',tail = 'upper'))
        '''
        rfesvm_split = SplitClassifier(LinearCSVMC())

        fsel = RFE(rfesvm_split.get_sensitivity_analyzer(
                        postproc=ChainMapper([FxMapper('features', l2_normed),
                                              FxMapper('samples', np.mean),
                                              FxMapper('samples', np.abs)])),
                  ConfusionBasedError(rfesvm_split, confusion_state='stats'),
                  Repeater(2),
                  fselector=FractionTailSelector(0.50,
                                                 mode='select',
                                                 tail='upper'),
                  stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),
                  train_pmeasure=False,
                  update_sensitivity=True)
        '''
        clf = LinearCSVMC(C=1, probability=1,
                          enable_ca=['probabilities', 'estimates'])

        #fclf = FeatureSelectionClassifier(clf, fsel)
        
        cv_storage = StoreResults()

        cvte = CrossValidation(clf,
                               NFoldPartitioner(cvtype=k, attr='band'),
                               callback=cv_storage,
                               enable_ca=['stats', 'repetition_results','raw_results'])
        
        err = np.mean(cvte(ds))

        err_lst.append([k, 1 - err])

        print('------------------------------------')
        print('n_folds = %d, n_feats = %d' %(k, n))
        print cvte.ca.stats

        sensana = fclf.get_sensitivity_analyzer()

        weights = sensana(ds)

        m = masked_mat[0]
        matrix = np.zeros_like(m)

        matrix[m!=0] = (weights.samples[0] - weights.samples[0].mean()) / weights.samples[0].std()

        thr_matrix = matrix * (np.abs(matrix) > 1.96)
        '''
        pl.figure(figsize=(13,13))
        pl.imshow(matrix, interpolation='nearest',cmap=pl.cm.seismic)
        pl.xticks(np.arange(len(labels)), labels, rotation='vertical')
        pl.yticks(np.arange(len(labels)), labels)
        pl.colorbar()
        #pl.savefig(os.path.join(datapath,'results','movie_weight_matrix_'+str(k)+'_folds_fsel.png'))
        
        pl.figure(figsize=(13,13))
        pl.imshow(thr_matrix, interpolation='nearest',cmap=pl.cm.seismic)
        pl.xticks(np.arange(len(labels)), labels, rotation='vertical')
        pl.yticks(np.arange(len(labels)), labels)
        pl.colorbar()
        #pl.savefig(os.path.join(datapath,'results','movie_weight_thresholded_matrix_'+str(k)+'_folds_fsel.png'))
        '''
        sens_mat.append(matrix)

def array_to_matrix(arr, dim, is_triangular=True):
    
    matrix = np.zeros((dim, dim))
    
    #This gives me indexes of lower part of matrix, diagonal included
    iu = np.triu_indices(dim)
    #This refines me indexes leaving diagonal indexes
    iu = (iu[0][iu[0] != iu[1]], iu[1][iu[0] != iu[1]])
    
    if is_triangular:
        matrix[iu] = arr
        matrix = copy_matrix(matrix)
    else:
        il = np.tril_indices(dim)
    
    return matrix
    


def copy_matrix(matrix):

    iu = np.triu_indices(matrix.shape[0])
    il = np.tril_indices(matrix.shape[0])

    matrix[il] = 1

    for i, j in zip(iu[0], iu[1]):
        matrix[j, i] = matrix[i, j]

    return matrix

def load_mat_dataset(datapath, bands, conditions, networks=None):
    
    target_list = []
    sample_list = []
    chunk_list = []
    band_list = []

    labels = np.loadtxt(os.path.join(datapath, 'roi_labels.txt'),
                    dtype=np.str_,
                    delimiter='\t')

    #labels = labels.T[1]
    subject_list_chunks = np.loadtxt(os.path.join(datapath, 'subj_list'),
                                 dtype=np.str)
    
    
    
    mask = np.zeros(len(labels.T[0]))
    if networks != None:
        for n in networks:
            mask += labels.T[-1] == n
        
    else:
        mask = np.ones(len(labels.T[0]), dtype=np.bool_)
    
    mask_roi = np.meshgrid(mask, mask)[1] * np.meshgrid(mask, mask)[0]
    
    for cond in conditions:
        for band in bands:
            filt_list = [f for f in filelist if f.find(cond) != -1 \
                                        and f.find(band) != -1]
            data = loadmat(os.path.join(datapath, filt_list[0]))

            mat_ = data[data.keys()[0]]

            #mat_[np.isinf(mat_)] = 0

            il = np.tril_indices(mat_[0].shape[0])

            masked_mat = mat_ * mask_roi[np.newaxis,:]

            for m in masked_mat:
                m[il] = 0

                #samples = np.array([m[il] = 0 for m in masked_mat])
            samples = np.array([m[np.nonzero(m)] for m in masked_mat])
            targets = [cond for i in samples]

            band_ = [band for i in samples]

            target_list.append(targets)
            sample_list.append(samples)
            chunk_list.append(subject_list_chunks)
            band_list.append(band_)

    targets = np.hstack(target_list)
    samples = np.vstack(sample_list)
    chunks = np.hstack(chunk_list)

    zsamples = sc_zscore(samples, axis=0)

    ds = dataset_wizard(zsamples, targets=targets, chunks=chunks)
    ds.sa['band'] = np.hstack(band_list)

    #zscore(ds)
        
    return ds

def plot_matrix():
    
    for cond in np.unique(targets):
        m_cond = targets == cond
        for band in np.unique(bands):
            m_band = bands == band
            
            total_m = m_cond * m_band

class StoreResults(object):
    def __init__(self):
        self.storage = []
            
    def __call__(self, data, node, result):
        #print node.measure
        #print data
        #print result
        self.storage.append((node.measure.ca.estimates,
                                    node.measure.ca.predictions))
        
        
######################################################
if __name__ == '__main__':
    
    datapath = '/home/robbis/data/corr_raw/RAW_mat_corr/'

    conditions = ['movie', 'scramble', 'rest']
    bands = ['alpha','beta','gamma','delta','theta']
    
    labels = np.loadtxt(os.path.join(datapath, 'roi_labels.txt'),
                    dtype=np.str_,
                    delimiter='\t')
    
    results = []
    
    for net in np.unique(labels.T[-1]):
        
        print '----- '+net+' -----'
        
        ds = load_mat_dataset(datapath, bands, conditions, networks=[net])
        
        for b in bands:
            
            ds_train = ds[(ds.targets != 'rest') * (ds.sa.band == b)]
            ds_test = ds[(ds.targets == 'rest') * (ds.sa.band == b)]
            
            clf = LinearCSVMC(C=1, probability=1,
                              enable_ca=['probabilities', 'estimates'])
            
            #clf = knn()
            
            cvte = CrossValidation(clf,
                                   NFoldPartitioner(cvtype=2, attr='chunks'),
                                   #callback=cv_storage,
                                   enable_ca=['stats', 'repetition_results','raw_results'])
            err = cvte(ds_train)
            
            print cvte.ca.stats
            
            prediction = clf.predict(ds_test)
            n_movie = np.count_nonzero(np.array(prediction) == 'movie')
            perc = n_movie/np.float(len(prediction))
            
            print perc
            
            results.append([net, b, 1 - np.mean(err), perc])
    