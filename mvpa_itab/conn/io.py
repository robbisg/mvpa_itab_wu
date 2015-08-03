import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore as sc_zscore
from mvpa2.suite import dataset_wizard, zscore
import os
from nitime.analysis.base import BaseAnalyzer
from scipy.spatial.distance import euclidean


def load_fcmri_dataset(data, subjects, conditions, group, level, n_run=3):
    
    attributes = []
    samples = []
    
    for ic, c in enumerate(conditions):
        for isb, s in enumerate(subjects):
            for i in range(n_run):
                      
                matrix = data[ic,isb,i,:]
                fmatrix = flatten_correlation_matrix(matrix)
                
                samples.append(fmatrix)
                attributes.append([c, s, i, group[isb], level[isb]])
    
    attributes = np.array(attributes)
    
    ds = dataset_wizard(np.array(samples), targets=attributes.T[0], chunks=attributes.T[1])
    ds.sa['run'] = attributes.T[2]
    ds.sa['group'] = attributes.T[3]
    ds.sa['meditation'] = attributes.T[0]
    ds.sa['level'] = np.int_(attributes.T[4])
    return ds
    


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
    #print filelist
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
    #zscore(ds, chunks_attr='chunks')
    #zscore(ds, chunks_attr='band')
    
    #print ds.shape
        
    return ds

def flatten_correlation_matrix(matrix):
    
    il = np.tril_indices(matrix.shape[0])
    out_matrix = matrix.copy()
    out_matrix[il] = np.nan
    
    out_matrix[range(matrix.shape[0]),range(matrix.shape[0])] = np.nan
    '''
    iu = np.triu_indices(matrix.shape[0])
    out_matrix[out_matrix[iu] == 0] = np.nan
    
    out_matrix = out_matrix[np.nonzero(out_matrix)]
    out_matrix[np.isnan(out_matrix)] == 0
    '''
    return matrix[~np.isnan(out_matrix)]

def copy_matrix(matrix):

    iu = np.triu_indices(matrix.shape[0])
    il = np.tril_indices(matrix.shape[0])

    matrix[il] = 1

    for i, j in zip(iu[0], iu[1]):
        matrix[j, i] = matrix[i, j]

    return matrix    

def load_correlation_matrix(path, pattern_):
    
    """
    Gets a pathname in which is supposed to be a list of txt files
    with the connectivity matrices, filenames are composed by a common
    pattern to filter the file list in the folder.
    
    The output is the array shaped subj x node x node
    """ 
    
    flist_conn = os.listdir(path)
    flist_conn = [f for f in flist_conn if f.find(pattern_) != -1]
        
    conn_data = []
    for f in flist_conn:
        data_ = np.genfromtxt(os.path.join(path, f))
        conn_data.append(data_)
    
    conn_data = np.array(conn_data)
    
    return conn_data

def load_correlation(path, filepattern, format, dictionary):
    """
    path: where to find the file?
    filepattern: is a single file or a serie of?
    format: which routine use to open the file?
    dictonary: what is the meaning of each matrix dimension?
    """
    # To be implemented
    
    return

class CorrelationLoader(object):
        
    def load(self, path, filepattern, conditions=None):
        
        # Check what we have in the path (subjdirs, subjfiles, singlefile)
        
        
        subjects = os.listdir(path)
            
        subjects = [s for s in subjects if s.find('configuration') == -1 \
            and s.find('.') == -1]
    
    
        result = []
    
        for c in conditions:

            s_list = []
    
            for s in subjects:

                sub_path = os.path.join(path, s)

                filel = os.listdir(sub_path)
                filel = [f for f in filel if f.find(c) != -1]
                c_list = []
                
                for f in filel:

                    matrix = np.loadtxt(os.path.join(sub_path, f))
            
                c_list.append(matrix)
    
            s_list.append(np.array(c_list))

            result.append(np.array(s_list))
    
        return np.array(result)   
        

class RegressionDataset(object):
    
    
    def __init__(self, X, y, group=None, conditions=None):
        
        self.X = X
        self.y = y
        
        if group != None:
            if len(self.group) != len(y):
                raise ValueError("Data mismatch: Check if \
                data and group have the same numerosity!")
            
        self.group = np.array(group)
        
        if conditions != None:
            if len(self.group) != len(y):
                raise ValueError("Data mismatch: Check if \
                data and conditions have the same numerosity!")
        
        self.conditions = conditions
    
    
    def get_group(self, group_name):
        
        if group_name not in np.unique(self.group):
            raise ValueError("%s not included in loaded groups!", 
                             group_name)
        
        group_mask = self.group == group_name
        
        rds = RegressionDataset(self.X[group_mask],
                                self.y[group_mask],
                                group=self.group[group_mask])
        return rds
    
    
       
        
            
               
    