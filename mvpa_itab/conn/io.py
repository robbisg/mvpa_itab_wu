import numpy as np
import nibabel as ni
from scipy.io import loadmat
from scipy.stats import zscore as sc_zscore
from mvpa2.suite import dataset_wizard, zscore
import os
from nitime.analysis.base import BaseAnalyzer
from nitime.timeseries import TimeSeries
from scipy.spatial.distance import euclidean
from mvpa2.datasets.base import Dataset, dataset_wizard
from mvpa_itab.conn.connectivity import load_matrices, z_fisher, glm, get_bold_signals
from mvpa_itab.conn.operations import flatten_correlation_matrix

import logging
logger = logging.getLogger(__name__)


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
    ds.sa['level'] = np.int_(attributes.T[4])
    ds.sa['meditation'] = attributes.T[0]

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
    


class ConnectivityLoader(object):
    
    def __init__(self, path, subjects, res_dir, roi_list):
        
        self.path = os.path.join(path, res_dir)
        self.subjects = subjects
        self.roi_list = roi_list
    
    
    def get_results(self, conditions):
        
        
        self.conditions = dict(zip(conditions, range(len(conditions))))
        
        # Loads data for each subject
        # results is in the form (condition x subjects x runs x matrix)
        results = load_matrices(self.path, conditions)
        
        # Check if there are NaNs in the data
        nan_mask = np.isnan(results)
        for _ in range(len(results.shape) - 2):
            # For each condition/subject/run check if we have nan
            nan_mask = nan_mask.sum(axis=0)
        
        
            
        
        #pl.imshow(np.bool_(nan_mask), interpolation='nearest')
        #print np.nonzero(np.bool_(nan_mask)[0,:])
        # Clean NaNs
        results = results[:,:,:,~np.bool_(nan_mask)]
        
        # Reshaping because numpy masking flattens matrices        
        rows = np.sqrt(results.shape[-1])
        shape = list(results.shape[:-1])
        shape.append(int(rows))
        shape.append(-1)
        results = results.reshape(shape)
        
        # We apply z fisher to results
        zresults = z_fisher(results)
        zresults[np.isinf(zresults)] = 1
        
        self.results = zresults
        
        # Select mask to delete labels
        roi_mask = ~np.bool_(np.diagonal(nan_mask))

        # Get some information to store stuff
        self.store_details(roi_mask)   

        # Mean across runs
        zmean = zresults.mean(axis=2)
                
        new_shape = list(zmean.shape[-2:])
        new_shape.insert(0, -1)
        
        zreshaped = zmean.reshape(new_shape)
        
        upper_mask = np.ones_like(zreshaped[0])
        upper_mask[np.tril_indices(zreshaped[0].shape[0])] = 0
        upper_mask = np.bool_(upper_mask)
        
        # Returns the mask of the not available ROIs.
        self.nan_mask = nan_mask
        return self.nan_mask


    def store_details(self, roi_mask):
        
        fields = dict()
        # Depending on data
        self.network_names = list(self.roi_list[roi_mask].T[0])
        #self.roi_names = list(self.roi_list[roi_mask].T[2]) #self.roi_names = list(self.roi_list[roi_mask].T[1])
        self.subject_groups = list(self.subjects.T[1])
        self.subject_level = list(np.int_(self.subjects.T[-1]))
        #self.networks = self.roi_list[roi_mask].T[-2]
        
        return fields


    def get_dataset(self):
        
        zresults = self.results
        
        new_shape = list(zresults.shape[-2:])
        new_shape.insert(0, -1)
        
        zreshaped = zresults.reshape(new_shape)
        
        upper_mask = np.ones_like(zreshaped[0])
        upper_mask[np.tril_indices(zreshaped[0].shape[0])] = 0
        upper_mask = np.bool_(upper_mask)
        
        # Reshape data to have samples x features
        ds_data = zreshaped[:,upper_mask]
    
        labels = []
        n_runs = zresults.shape[2]
        n_subj = zresults.shape[1]
        
        for l in self.conditions.keys():
            labels += [l for _ in range(n_runs * n_subj)]
        ds_labels = np.array(labels)
        
        ds_subjects = []

        for s in self.subjects:
            ds_subjects += [s for _ in range(n_runs)]
        ds_subjects = np.array(ds_subjects)
        
        ds_info = []
        for _ in self.conditions.keys():
            ds_info.append(ds_subjects)
        ds_info = np.vstack(ds_info)
        
        
        self.ds = dataset_wizard(ds_data, targets=ds_labels, chunks=np.int_(ds_info.T[5]))
        self.ds.sa['subjects'] = ds_info.T[0]
        self.ds.sa['groups'] = ds_info.T[1]
        self.ds.sa['chunks_1'] = ds_info.T[2]
        self.ds.sa['expertise'] = ds_info.T[3]
        self.ds.sa['age'] = ds_info.T[4]
        self.ds.sa['chunks_2'] = ds_info.T[5]
        self.ds.sa['meditation'] = ds_labels
        
        logger.debug(ds_info.T[4])
        logger.debug(self.ds.sa.keys())
        
        return self.ds
              


class ConnectivityPreprocessing(object):
    
    def __init__(self, path, subject, boldfile, brainmask, regressormask, subdir='fmri'):
        
        self.path = path
        self.subject = subject
        self.subdir = subdir
        self.bold = ni.load(os.path.join(path, subject, subdir, boldfile))
        self.loadedSignals = False
        self.brain_mask = ni.load(os.path.join(path, subject, subdir, brainmask))
        
        self.mask = []
        for mask_ in regressormask:
            m = ni.load(os.path.join(path, subject, subdir, mask_))
            self.mask.append(m)
                    
    
    def execute(self, gsr=True, filter_params={'ub': 0.08, 'lb':0.009}, tr=4.):
        
        # Get timeseries
        if not self.loadedSignals:
            self._load_signals(tr, gsr, filter_params=filter_params)
        elif self.loadedSignals['gsr']!=gsr or self.loadedSignals['filter_params']!=filter_params:
            self._load_signals(tr, gsr, filter_params=filter_params)
        
        beta = glm(self.fmri_ts.data.T, self.regressors.T)
        
        residuals = self.fmri_ts.data.T - np.dot(self.regressors.T, beta)
        
        ts_residual = TimeSeries(residuals.T, sampling_interval=tr)
    
        '''
        ub = filter_params['ub']
        lb = filter_params['lb']
        
        F = FilterAnalyzer(ts_residual, ub=ub, lb=lb)
        '''
        residual_4d = np.zeros_like(self.bold.get_data())
        residual_4d [self.brain_mask.get_data() > 0] = ts_residual.data
        residual_4d[np.isnan(residual_4d)] = 0
        
        self._save(residual_4d, gsr=gsr)
        
    
    def _save(self, image, gsr=True):
        
        gsr_string = ''
        if gsr:
            gsr_string = '_gsr'
        
        filename = 'residual_filtered_first%s.nii.gz' % (gsr_string)
        img = ni.Nifti1Image(image, self.bold.get_affine())
        filepath = os.path.join(self.path, self.subject, self.subdir, filename)
        
        ni.save(img, filepath)
        
    
    
    def _load_signals(self, tr, gsr, filter_params=None):
        
        regressor = []
        
        self.fmri_ts = get_bold_signals(self.bold, 
                                        self.brain_mask, 
                                        tr, 
                                        ts_extraction='none',
                                        filter_par=filter_params)
        
        if gsr:
            gsr_ts = get_bold_signals(self.bold, 
                                      self.brain_mask, 
                                      tr, 
                                      filter_par=filter_params)
            regressor.append(gsr_ts.data)
        
        for mask_ in self.mask:
            ts_ = get_bold_signals(self.bold, 
                                   mask_, 
                                   tr,
                                   filter_par=filter_params ) 
            regressor.append(ts_.data)
        
        self.loadedSignals = {'gsr':gsr, 'filter_params':filter_params}
        self.regressors = np.vstack(regressor)      
            
               
    