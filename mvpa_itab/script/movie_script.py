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
from nitime import timeseries
from nitime.timeseries import TimeSeries
from nitime.analysis.correlation import SeedCorrelationAnalyzer
from scipy.stats.stats import ttest_ind

import matplotlib.pyplot as pl
from mvpa_itab.similarity import SeedCorrelationAnalyzerWrapper,\
    SeedSimilarityAnalysis, SeedAnalyzer
from scipy.spatial.distance import euclidean
from mvpa_itab.measure import mutual_information
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
                                       th and f.find(band) != -1]
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
    datapath = '/media/DATA/fmri/movie_viviana/corr_raw/RAW_mat_corr/'
    
    conditions = ['movie', 'scramble', 'rest']
    bands = ['alpha','beta','gamma','delta','theta']
    
    labels = np.loadtxt(os.path.join(datapath, 'roi_labels.txt'),
                    dtype=np.str_,
                    delimiter='\t')
    
    results = []
    
    for net in np.unique(labels.T[-1])[:1]:
        
        print '----- '+net+' -----'
        
        ds = load_mat_dataset(datapath, bands, conditions)#, networks=[net])
        
        for b in bands:
            
            #ds_train = ds[(ds.targets != 'rest') * (ds.sa.band == b)]
            #ds_test = ds[(ds.targets == 'rest') * (ds.sa.band == b)]
            
            dist = permutations(ds[ds.sa.band == b], n_permutations=1501)
            
            results.append([net, b, dist])
            ###################################################################
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
############################################
path = '/home/robbis/Share/Vivi/'
mat_list = os.listdir('/home/robbis/Share/Vivi/')
mat_list = [f for f in mat_list if f.find('.mat') != -1]
data = dict()

mask_mat = np.ones((50,50))
mask_mat[np.tril_indices(50)] = 0
mat_list.sort()
roi_path = '/media/robbis/DATA/fmri/movie_viviana/corr_raw/RAW_mat_corr'
roi_mat = np.zeros((50,50))
roi_labels = np.loadtxt(os.path.join(roi_path, 'roi_labels.txt'),
                    dtype=np.str_,
                    delimiter='\t')

# re-ordering roi_labels vector thus np.unique alphabetically order the vector
unique_labels, indexes = np.unique(roi_labels.T[-1], return_index=True)
ord_ind = np.argsort(np.argsort(unique_labels))

# Building a matrix mask, with a number identifying each ROI
for i, n in enumerate(np.unique(roi_labels.T[-1])):
    mask = roi_labels.T[-1] == n # Vector mask
    mask_roi = np.meshgrid(mask, mask)[1] * np.meshgrid(mask, mask)[0] # Matrix mask
    roi_mat = mask_roi*(ord_ind[i]+1) + roi_mat # The ROI index is put in the matrix mask

index = 0

for f in mat_list[:3]:
    # Load matrix
    mat_file = loadmat(os.path.join(path, f))
    # String stuff to find dictionary label
    key = f[:f.find('1')]
    klist = key.split('_')
    condition = klist[-1]
    klist.pop()
    klist.pop()
    klist.append(condition)
    key = '_'.join(klist)
    
    ts_matrix = mat_file[key][:,:,np.bool_(mask_mat)]
    #ts_matrix = sc_zscore(ts_matrix, axis=0)
    if index != 0:
        roi_mask_mat = mask_mat * (roi_mat == index)
    else:
        roi_mask_mat = mask_mat
    print mat_file[key].shape
    data[str.lower(condition)] = mat_file[key][:,:,np.bool_(roi_mask_mat)]
    
data_movie = data['movie']
data_rest = data['rest']


seed_analyzer = SeedAnalyzer
kwargs = {'measure': euclidean}

#kwargs = {'measure': mutual_information}


seed_analyzer = SeedCorrelationAnalyzerWrapper

for i in range(data_movie.shape[1]):
    seed_ds = data_rest[:,i,:]
    #target_ds = data_movie[i]
    seed_similarity = SeedSimilarityAnalysis(seed_ds=seed_ds,
                                             seed_analyzer=seed_analyzer,
                                             **kwargs)
    
    value = seed_similarity.run(target_ds)
    perm = seed_similarity.permutation_test(target_ds, 
                                            permutation=1000,
                                            axis=1)
    p = seed_similarity.p_values(value)
    f = pl.figure()
    pl
    
    
    



"""
ts = dict()
correlation = []
for i in range(11):
    for k in data.keys():
        ts[k] = TimeSeries(data[k][i,:], sampling_interval=1.)
    
    C1 = SeedCorrelationAnalyzer(ts['rest'], ts['movie'])
    C2 = SeedCorrelationAnalyzer(ts['rest'], ts['scramble'])
    
    correlation.append([C1.corrcoef, C2.corrcoef])

for c in correlation:
    c[0] = c[0][:c[1].shape[0], :c[1].shape[1]]

corr = np.array(correlation)
"""



#Plots
mat_movie_time = loadmat('/home/robbis/Share/Vivi/time_corr_movie1.mat')
mat_rest_time = loadmat('/home/robbis/Share/Vivi/time_corr_rest.mat')
mat_scramble_time = loadmat('/home/robbis/Share/Vivi/time_corr_scramble1.mat')

key = 'time_corr'
time_movie = mat_movie_time[key]
time_rest = mat_rest_time[key]
time_scramble = mat_scramble_time[key]

trail_ = str.lower(un_vec[np.argsort(ind)][index-1])

path__ = '/media/robbis/DATA/fmri/movie_viviana/similarity_results/'+trail_
for i in range(11):
    pl.figure()
    pl.imshow(corr[i,0,...], vmax=1, vmin=-1)
    pl.colorbar()
    pl.savefig(os.path.join(path__, str(i+1)+'_sbj_movie_'+trail_+'.png'), dpi=150)
    pl.figure()
    pl.imshow(corr[i,1,...], vmax=1, vmin=-1)
    pl.colorbar()
    pl.savefig(os.path.join(path__, str(i+1)+'_sbj_scramble_'+trail_+'.png'), dpi=150)

pl.close('all')

pl.figure()
for i in range(11):
    pl.plot(corr[i,0,...].mean(0), c='r', alpha=0.2)
    pl.plot(corr[i,1,...].mean(0), c='b', alpha=0.2)
pl.plot(corr[:,0,...].mean(1).mean(0), c='r', linewidth=1.5, label='movie')
pl.plot(corr[:,1,...].mean(1).mean(0), c='b', linewidth=1.5, label='scramble')
pl.legend()
pl.savefig(os.path.join(path__, 'plot_movie_scramble_'+trail_+'.png'), dpi=300)

