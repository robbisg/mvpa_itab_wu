# pylint: disable=maybe-no-member, method-hidden
import os
import nibabel as ni
import numpy as np
from mvpa_itab.connectivity import glm, get_bold_signals, load_matrices, z_fisher
from nitime.timeseries import TimeSeries
import itertools


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



def find_roi_center(img, roi_value):
    
    affine = img.get_affine()
    
    mask_ = np.int_(img.get_data()) == roi_value
    ijk_coords = np.array(np.nonzero(mask_)).mean(1)
    
    xyz_coords = ijk_coords * affine.diagonal()[:-1] + affine[:-1,-1]
    
    return xyz_coords



def get_atlas90_coords():
    atlas90 = ni.load('/media/robbis/DATA/fmri/templates_AAL/atlas90_mni_2mm.nii.gz')
    coords = [find_roi_center(atlas90, roi_value=i) for i in np.unique(atlas90.get_data())[1:]]
    
    return np.array(coords)



def get_findlab_coords():
    roi_list = os.listdir('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/')
    roi_list.sort()
    findlab = [ni.load('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/'+roi) for roi in roi_list]
    f_coords = []
    for img_ in findlab:
        f_coords.append(np.array([find_roi_center(img_, roi_value=np.int(i)) for i in np.unique(img_.get_data())[1:]]))
        
    return np.vstack(f_coords)
           

def array_to_matrix(array, nan_mask=None):
    
    if nan_mask == None:
        # second degree resolution to get matrix dimensions #
        c = -2*array.shape[0]
        a = 1
        b = -1
        det =  b*b - 4*a*c
        rows = np.int((-b + np.sqrt(det))/(2*a))
        
        matrix = np.ones((rows, rows))
    else:
        matrix = np.float_(np.logical_not(nan_mask))
    
    il = np.tril_indices(matrix.shape[0])
    matrix[il] = 0
    
    matrix[np.nonzero(matrix)] = array
    
    return matrix


def get_plot_stuff(directory_):
    
    if directory_.find('atlas90') != -1 or directory_.find('20150') != -1:
        coords = get_atlas90_coords()
        roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_AAL/atlas90.cod',
                              delimiter='=',
                              dtype=np.str)
        names = roi_list.T[1]
        names_inv = np.array([n[::-1] for n in names])
        index_ = np.argsort(names_inv)
        names_lr = names[index_]
        dict_ = {'L':'#89CC74', 'R':'#7A84CC'}
        colors_lr = np.array([dict_[n[:1]] for n in names_inv])    
        names = np.array([n.replace('_', ' ') for n in names])

    
    elif directory_.find('findlab') != -1 or directory_.find('2014') != -1:
        coords = get_findlab_coords()
        roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)
        names = roi_list.T[2]

        dict_ = {'Auditory':'silver', 
                 'Basal_Ganglia':'white', 
                 'LECN':'red',
                 'Language':'orange', 
                 'Precuneus':'green',
                 'RECN':'plum', 
                 'Sensorimotor':'gold', 
                 'Visuospatial':'blueviolet', 
                 'anterior_Salience':'beige',
                 'dorsal_DMN':'cyan', 
                 'high_Visual':'yellow', 
                 'post_Salience':'lime', 
                 'prim_Visual':'magenta',
                 'ventral_DMN':'royalblue'
                 }
        
        colors_lr = np.array([dict_[r.T[-2]] for r in roi_list])
        index_ = np.arange(90)
        
        
    return names, colors_lr, index_, coords


def flatten_correlation_matrix(matrix):
    
    il = np.tril_indices(matrix.shape[0])
    out_matrix = matrix.copy()
    out_matrix[il] = np.nan
    
    out_matrix[range(matrix.shape[0]),range(matrix.shape[0])] = np.nan

    return matrix[~np.isnan(out_matrix)]



def copy_matrix(matrix, diagonal_filler=1):

    iu = np.triu_indices(matrix.shape[0])
    il = np.tril_indices(matrix.shape[0])

    matrix[il] = diagonal_filler

    for i, j in zip(iu[0], iu[1]):
        matrix[j, i] = matrix[i, j]

    return matrix    
  

        
def network_connections(matrix, label, roi_list, method='within'):
    """
    Function used to extract within- or between-networks values
    """
    
    mask1 = roi_list == label
    
    if method == 'within':
        mask2 = roi_list == label
    else:
        mask2 = roi_list != label
    
    matrix_mask = np.meshgrid(mask1, mask1)[1] * np.meshgrid(mask2, mask2)[0]
    
    connections_ = matrix * matrix_mask
    
    return connections_
    


def get_signed_connectome(matrix, method='negative'):
    """
    Function used to extract positive or negative values from matrix
    """
    
    sign = 1
    if method == 'negative':
        sign = -1
    
    mask_ = (matrix * sign) > 0
    signed_matrix = matrix * mask_
    
    return signed_matrix       


def aggregate_networks(matrix, roi_list):
    """
    Function used to aggregate matrix values using 
    aggregative information provided by roi_list
    """
    
    unique_rois = np.unique(roi_list)
    n_roi = unique_rois.shape[0]

    aggregate_matrix = np.zeros((n_roi, n_roi), dtype=np.float)
    
    network_pairs = itertools.combinations(unique_rois, 2)
    indexes = np.vstack(np.triu_indices(n_roi, k=1)).T
    
    # This is to fill upper part of the aggregate matrix
    for i, (n1, n2) in enumerate(network_pairs):
        
        x = indexes[i][0]
        y = indexes[i][1]
        
        mask1 = roi_list == n1
        mask2 = roi_list == n2
        
        # Build the mask of the intersection between
        mask_roi = np.meshgrid(mask1, mask1)[1] * np.meshgrid(mask2, mask2)[0]
        
        value = np.sum(matrix * mask_roi)
        
        aggregate_matrix[x, y] = value
    
    # Copy matrix in the lower part
    aggregate_matrix = copy_matrix(aggregate_matrix)
    
    # This is to fill the diagonal with within-network sum of elements
    for i, n in enumerate(unique_rois):
        
        diag_matrix = network_connections(matrix, n, roi_list)
        aggregate_matrix[i, i] = np.sum(diag_matrix)
        
    
    return aggregate_matrix
        
        
        
        
        
