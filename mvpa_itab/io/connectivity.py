import os
import numpy as np
from mvpa2.datasets.base import dataset_wizard, Dataset
from pyitab.io.subjects import load_subject_file, add_subjectname
from pyitab.io.base import load_attributes, load_filelist
from mvpa_itab.regression.base import Analysis
from mvpa_itab.conn.connectivity import z_fisher
from pyitab.base import Node
from pyitab.io.configuration import read_configuration
from mvpa2.base.dataset import vstack

import logging
from mvpa2.base.collections import SampleAttributesCollection
from sklearn.preprocessing.label import LabelEncoder
logger = logging.getLogger(__name__)



def conn_transformer(data):
    
    data = np.rollaxis(np.vstack(data), 0 ,3)
    data = data[np.triu_indices(data.shape[0], k=1)].T
    logger.debug(data.shape)
    
    return data



def load_meg_seed_ds(conf_file, task, prepro=Node(), **kwargs):
    
    # TODO: conf file should include the full path
    conf = read_configuration(conf_file, task)
           
    conf.update(kwargs)
    logger.debug(conf)
    
    data_path = conf['data_path']
    
    # Subject file should be included in configuration
    subject_file = conf['subjects']
    subjects, extra_sa = load_subject_file(subject_file)
        
    logger.info('Merging %s subjects from %s' % (str(len(subjects)), data_path))
    
    for i, subj in enumerate(subjects):
        
        ds = load_mat_ds(data_path, subj, task, **conf)
        
        ds = prepro.transform(ds)
        
        logger.debug(ds.shape)
        
        # add extra samples
        if extra_sa != None:
            for k, v in extra_sa.iteritems():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
                    
        # First subject
        if i == 0:
            ds_merged = ds.copy()
        else:
            ds_merged = vstack((ds_merged, ds))
            ds_merged.a.update(ds.a)
            
        
        del ds
    
    ds_merged.a['prepro'] = prepro.get_names()
    
    return ds_merged





def load_mat_data(path, subj, folder, **kwargs):
    
    from scipy.io import loadmat
    
    meg_transformer = {'connectivity': conn_transformer,
                       'power': lambda data: np.vstack(data)}
    
    key = kwargs['mat_key']
    transformer = meg_transformer[kwargs['transformer']]
    
    # load data from mat
    filelist = load_filelist(path, subj, folder, **kwargs)
    
    data = []
    for f in filelist:
        
        logger.info("Loading %s..." %(f))
        mat = loadmat(f)
        datum = mat[key]
        logger.debug(datum.shape)
        data.append(mat[key])
    
    return transformer(data)




def load_mat_ds(path, subj, folder, **kwargs):   
    
        
    data = load_mat_data(path, subj, folder, **kwargs)
    
    # load attributes
    attr = load_attributes(path, subj, folder, **kwargs)
    
    attr, labels = edit_attr(attr, data.shape)
    
    
    ds = Dataset.from_wizard(data, attr.targets)
    ds = add_subjectname(ds, subj)
    ds = add_attributes(ds, attr)
    
    #ds.fa['roi_labels'] = labels
    ds.fa['matrix_values'] = np.ones_like(data[0])
    
    ds.sa['chunks'] = LabelEncoder().fit_transform(ds.sa['name'])
    
    return ds




def edit_attr(attr, shape):
        
    factor = shape[0]/len(attr.targets)

    attr_ = dict()
    for key in attr.keys():
        attr_[key] = []
        for label in attr[key]:
            attr_[key] += [label for i in range(factor)]
            
    """    
    attr_['roi_labels'] = []
    for j in range(len(attr.targets)):
        for i in range(shape[1]):
            attr_['roi_labels'] += ["roi_%02d" % (i+1)]
    """
    
       
    return SampleAttributesCollection(attr_), None#attr_['roi_labels'][:shape[1]]
    



def load_connectivity_ds(path, subjects, extra_sa_file=None):
    """
    Loads txt connectivity matrices in the form of n_roi x n_roi matrix
    
    Parameters
    ----------
    path : string
        Pathname of the data folder
        
    subjects : list
        List of subjects included in the analysis
        it is always a directory in the path
        
    extra_sa : dictionary
        Dictionary of extra fields to be included.
        The dictionary must be in the form of 
            'field': list of extra fields
        dictionary must include the field 'subject'
        
    Returns
    -------
    ds : pympva dataset
        The loaded dataset
    """
    
    data, attributes = load_txt_matrices(path, subjects)
    
    n_roi = data.shape[1]
    indices = np.triu_indices(n_roi, k=1)
    
    
    n_samples = data.shape[0]
    n_features = len(indices[0])
    samples = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        samples[i] = data[i][indices]
        
    mask = np.isnan(samples).sum(0)
    
    
    ds = dataset_wizard(samples)
    for key, value in attributes.iteritems():
        ds.sa[key] = value
        
    ds.fa['nan_mask'] = np.bool_(mask)
    
    if extra_sa_file != None:
        _, fextra_sa = load_subject_file(extra_sa_file)
        ds = add_subject_attributes(ds, fextra_sa)
    
    return ds



def add_subject_attributes(ds, extra_sa, ds_key='subject'):
    """
    ds_key is a key that should be shared 
    in extra_sa keys and ds sample attributes
    """
    
    
    attributes = {k:[] for k in extra_sa.keys() if k != ds_key}
    
    
    for sample in ds:
        for key, value in extra_sa.iteritems():
            if key == ds_key:
                continue
            
            v = value[extra_sa[ds_key] == sample.sa[ds_key].value][0]
            attributes[key].append(v)
            
    
    ds = add_attributes(ds, attributes)
    
    return ds
            
                


def load_txt_matrices(path, subjects):
    """
    path = path of result file
    conditions = analysis label
    """
        

    results = []

    attributes = {'condition':[], 'run':[], 'subject':[]}
    
    for s in subjects:
    
        path_data = os.path.join(path, s)
        matrix_list = os.listdir(path_data)
                
        for mat in matrix_list:
            
            matrix = np.loadtxt(os.path.join(path_data, mat))
            
            subj_attributes = get_txt_metadata(mat)
            
            for k, v in subj_attributes.items():
                if k in attributes.keys():
                    attributes[k].append(v)
                else:
                    attributes[k] = [v]
                    
            attributes['subject'].append(s)
        
            results.append(matrix)
            
    return np.array(results), attributes
    
    
def get_txt_metadata(filename):
    
    """
    filename should be in the form
    correlation_cond_condition1_run_run1_field_fielddata1.txt or
    correlation_condition1_run_run1_field_fielddata1.txt
    """
    
    fieldlist = filename[:-4].split("_")
    
    return {'condition':fieldlist[1], 'run':fieldlist[3]}
        


def load_matrices(path, condition):
    """
    path = path of result file
    conditions = analysis label
    """
    
    # Why looking for subjects in this way???
    subjects = os.listdir(path)
    
    subjects = [s for s in subjects if s.find('configuration') == -1 \
                and s.find('.') == -1 ]
    subjects = [s for s in subjects if s.find("expertise") == -1]
    
    logger.debug(path)
    result = []
    logger.debug(subjects)
    
    # Why filter here?
    for c in condition:

        s_list = []
        
        for s in subjects:

            sub_path = os.path.join(path, s)

            filel = os.listdir(sub_path)
            filel = [f for f in filel if f.find(c) != -1]
            c_list = []
            
            for f in filel:
                logger.debug(s+" "+f)
                matrix = np.loadtxt(os.path.join(sub_path, f))
                
                c_list.append(matrix)
        
            s_list.append(np.array(c_list))
    
        result.append(np.array(s_list))
        
    return np.array(result)



def load_meg_seed_data(path):
    return


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
        
        logger.debug(ds_info)
        logger.debug(ds_info.shape)
        
        logger.debug(ds_data.shape)
        
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



class ConnectivityDataLoader(Analysis):
    
    def setup_analysis(self, path, roi_list, directory, conditions, subjects):
        
        self.directory = directory
        self.conditions = conditions
        self.subjects = subjects
        
        conn = ConnectivityLoader(path, self.subjects, self.directory, roi_list)
        logger.debug(self.conditions)
        conn.get_results(self.conditions)
        self.ds = conn.get_dataset()
        
        return self

    
    def filter(self, filter_, operation='and'):
        """Filter is a dictionary the key is the sample attribute 
        the value is the value
        to filter, if the dictionary has more than one item a 
        logical and is performed"""
        
        if operation == 'and':
            func = np.logical_and
            mask = np.ones_like(self.ds.targets, dtype=np.bool)
        else:
            func = np.logical_or
            mask = np.zeros_like(self.ds.targets, dtype=np.bool)
            
        for k, v in filter_.items():
            logger.debug(mask)
            logger.debug(k)
            logger.debug(v)
            mask = func(mask, self.ds.sa[k].value == v)
            
        self.ds = self.ds[mask]
        
        logger.debug(self.ds.shape)
        
        return self
    
    
    def get_data(self, y_field='expertise'):
        """
        Get X matrix of samples and y array of outcomes
        """
        logger.debug(self.ds.sa['age'])
        logger.debug(y_field)
        return self.ds.samples, self.ds.sa[y_field].value 