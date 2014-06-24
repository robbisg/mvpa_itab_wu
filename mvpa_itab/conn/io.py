import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore as sc_zscore
from mvpa2.suite import dataset_wizard, zscore
import os



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
    
    filelist = os.listdir(datapath)
    filelist = [f for f in filelist if f.find('.mat') != -1]
    
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

    #zsamples = sc_zscore(samples, axis=1)

    ds = dataset_wizard(samples, targets=targets, chunks=chunks)
    ds.sa['band'] = np.hstack(band_list)
    
    #zscore(ds, chunks_attr='band')
    zscore(ds, chunks_attr='chunks')
    #zscore(ds, chunks_attr='band')
    
    print ds.shape
        
    return ds

def load_correlation():
    # To be implemented
    
    
    
    return