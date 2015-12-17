# pylint: disable=maybe-no-member, method-hidden
import os
import nibabel as ni
import numpy as np
from mvpa_itab.connectivity import glm, get_bold_signals, load_matrices, z_fisher
from nitime.timeseries import TimeSeries
from mvpa2.datasets.base import Dataset, dataset_wizard


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


class ConnectivityTest(object):
    
    def __init__(self, path, subjects, res_dir, roi_list):
        
        self.path = os.path.join(path, '0_results', res_dir)
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
        ds_info = np.vstack((ds_subjects, ds_subjects))
        
        
        self.ds = dataset_wizard(ds_data, targets=ds_labels, chunks=np.int_(ds_info.T[4]))
        self.ds.sa['subjects'] = ds_info.T[0]
        self.ds.sa['groups'] = ds_info.T[1]
        self.ds.sa['expertise'] = ds_info.T[3]
        self.ds.sa['chunks_'] = ds_info.T[2]
        self.ds.sa['meditation'] = ds_labels
        
        return self.ds
        
        
           
        
        
        
