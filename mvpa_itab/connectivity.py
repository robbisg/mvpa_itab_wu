# pylint: disable=maybe-no-member, method-hidden
import nitime.analysis as nta
import nitime.fmri.io as io

from nipy.modalities.fmri.glm import GeneralLinearModel

from nitime.timeseries import TimeSeries
from nitime.analysis.correlation import CorrelationAnalyzer,\
    SeedCorrelationAnalyzer
from mvpa_itab.io.base import get_time
from scipy.signal.windows import boxcar
from scipy.signal.signaltools import convolve
from scipy.stats.mstats import zscore

from mvpa_itab.io.base import load_wu_file_list, read_configuration, get_time

import matplotlib.pyplot as pl

import os
import nibabel as ni
import numpy as np
from nitime.analysis.coherence import CoherenceAnalyzer

from conn.plot import *
from nitime.analysis.spectral import FilterAnalyzer

from memory_profiler import profile
from scipy.stats.stats import ttest_ind

from sklearn.decomposition import PCA
from docutils.parsers.rst.directives import path
from oauthlib.oauth1.rfc5849.utils import filter_params


#@profile
def analyze_connectivity(imagelist, path_roi, roi_names, ts_param, **kwargs):
    
    TR = 1.
    for arg in kwargs:
        if arg == 'TR':
            TR = np.float(kwargs[arg])
    
    roi_list = os.listdir(path_roi)
    roi_list = [r for r in roi_list if r.find('.hdr') != -1 \
                                    or r.find('nii.gz') != -1]
    '''
    roi_list = ['lower_left_new_.nii.gz',
               'lower_right_new_.nii.gz',
               'upper_left_new_.nii.gz',
               'upper_right_new_.nii.gz',
               'roi_attention_all.4dfp.hdr',
               'roi_default_all.4dfp.hdr',
               'searchlight_separated.nii.gz'
               ]
    '''
    #roi_names = np.loadtxt('/media/DATA/fmri/learning/roi_labels', dtype=np.str)
    print 'Length of image list is '+str(len(imagelist))
    volume_shape = imagelist[0].get_data().shape[:-1]
    n_shape = list(volume_shape)
    n_shape.append(-1)
    coords = list(np.ndindex(volume_shape))
    coords = np.array(coords).reshape(n_shape)
    
    data_arr = []
    for roi in roi_list:
        #print roi
        r_mask = ni.load(os.path.join(path_roi, roi))
        mask = r_mask.get_data().squeeze()
        roi_filt = roi_names[roi_names.T[1] == roi]
        for label in np.unique(mask)[1:]:
            
            roi_m = np.int_(roi_filt.T[2]) == label
            if roi_m.any():
                print 'Loading voxels from '+roi_filt[roi_m].T[0][0]
                time_serie = io.time_series_from_file([f.get_filename() for f in imagelist], \
                                                      coords[mask==label].T, \
                                                      TR=float(TR), \
                                                      normalize=ts_param['normalize'], \
                                                      average=ts_param['average'], \
                                                      filter=ts_param['filter'])
                data_arr.append(time_serie)
        del r_mask
    
    data = np.array(data_arr)
    ts = TimeSeries(data, sampling_interval=float(TR))
    del imagelist, time_serie
    C = nta.CorrelationAnalyzer(ts)
    
    return C, ts


def create_similarity_files(path, subjects, target_label, source_label):
    
    
    for s in subjects:
        for cond in ['trained', 'untrained']:
            for time in ['RestPre', 'RestPost']:
                data_list = []
                roi_list = []
                for folder in folder_list:
                    roi = folder.split('_')[-2]
                    path_file = os.path.join(path, '0_results', folder, s)
                    data = np.loadtxt(os.path.join(path_file,
                                                    '_'.join([s,'distance','txt',cond,time,'.txt'])))
                    
                    data_list.append(data)
                    roi_list.append(roi)
                
                data_a = np.vstack(data_list).T
                roi_a = np.array(roi_list)
                
                path_save = os.path.join(path, '0_results', 'connectivity', 'similarity')
                
                np.savetxt(os.path.join(path_save, '_'.join(['similarity',cond,time,s,'.txt'])), data_a, 
                           fmt='%4.4f', delimiter=',')
    np.savetxt(os.path.join(path_save, 'roi_labels.txt'), roi_a, fmt='%s')                      


def bold_convolution(bold_timeseries, duration, win_func=boxcar):
              
    window=win_func(duration)
    
    n_roi = bold_timeseries.data.shape[0]
    convolved_bold = []
    for i in range(n_roi):
        convolution = convolve(np.abs(bold_timeseries.data[i]), window)
        convolved_bold.append(convolution)
    
    ts_convolved = TimeSeries(np.vstack(convolved_bold), 
                              sampling_interval=np.float(bold_timeseries.sampling_interval))
    
    return ts_convolved



 
def remove_bold_effect(bold_ts, distance_ts, ts_param, **kwargs):  
    
    for arg in kwargs:
        if arg == 'runs':
            n_runs = np.int(kwargs[arg])
        if arg == 'tr':
            TR = np.float(kwargs[arg])
    
    n_roi = bold_ts.data.shape[0]
 
    rl = bold_ts.data.shape[1]/n_runs - 1 #Run Length (minus one is deconvolution effect)
    dl = distance_ts.data.shape[1]/n_runs #Distance Length
    
    diff = rl - dl
       
    print distance_ts.data.shape
    deconv_ts = []
    for i in range(n_roi):
        deconv_distance = []
        for j in range(n_runs):
            full_data = np.ones(rl)
            full_data[diff:] = distance_ts.data[i][j*dl:(j+1)*dl]
            n_data = full_data/(bold_ts.data[i][j*rl:(j+1)*rl])
            deconv_distance.append(n_data[diff:])
        
        assert np.hstack(deconv_distance).shape[0] == dl*n_runs
        deconv_ts.append(np.hstack(deconv_distance))
    
    ts_deconv = TimeSeries(np.vstack(deconv_ts), sampling_interval=TR)
    
    
    return ts_deconv


def get_bold_timeserie(imagelist, path_roi, roi_names, ts_param, detrend=False, **kwargs):
    '''
    To be modified
    
    Enhancement: 
        - create a file/class for the rois with all the informations about it
                (network, long_name, name, pathfile, mask_value)
        - figure out how to 
                
    '''
     
    
    TR = 1.
    for arg in kwargs:
        if arg == 'tr':
            TR = np.float(kwargs[arg])
    
    #print TR            
    #roi_list = os.listdir(path_roi)
    #roi_list = [r for r in roi_list if r.find('.hdr') != -1 or r.find('nii.gz') != -1]
    
    print 'Length of image list is '+str(len(imagelist))
    volume_shape = imagelist[0].get_data().shape[:-1]
    n_shape = list(volume_shape)
    n_shape.append(-1)
    coords = list(np.ndindex(volume_shape))
    coords = np.array(coords).reshape(n_shape)
    n_runs = len(imagelist)
    
    data_arr = []
    for roi in np.unique(roi_names.T[-2]):
        #print roi
        # roi_names.T[1][i]
        roi_name = roi
        #print roi_name
        #network_name = roi[0]
        r_mask = ni.load(os.path.join(path_roi, roi_name, roi_name+'_separated_3mm.nii.gz'))
        mask = r_mask.get_data().squeeze()
        roi_filt = roi_names[roi_names.T[-2] == roi_name]
        #print roi_filt
        for label in np.int_(np.unique(mask)[1:]):
            roi_m = np.int_(roi_filt.T[-1]) == label
            
            if roi_m.any():
                print 'Loading voxels from '+roi_filt[roi_m].T[2][0]
                time_serie = io.time_series_from_file([f.get_filename() for f in imagelist], \
                                                      coords[mask==label].T, \
                                                      TR=float(TR), \
                                                      normalize=ts_param['normalize'], \
                                                      average=ts_param['average'], \
                                                      filter=ts_param['filter'])
                
                data_new = []
                
                #print time_serie.data.shape
                if detrend == True:
                    i = 0
                    for vt in time_serie.data:
                        run_split = np.split(vt, n_runs)
                        i = i + 1
                        #Detrending                   
                        d_vt = np.hstack([detrend(r) for r in run_split])
                    
                        z_vt = (d_vt - np.mean(d_vt))/np.std(d_vt)
                    
                        z_vt[np.isnan(z_vt)] = 0
                        data_new.append(z_vt)
                        
                    time_serie.data = np.vstack(data_new)
                    
                '''
                if ts_param['average'] == True:
                    ts_new = np.mean(time_serie.data, axis=0)
                '''   
                
                data_arr.append(time_serie.data)

        del r_mask
    
    data = np.vstack(data_arr)
    data[np.isnan(data)] = 0
    ts = TimeSeries(data, sampling_interval=float(TR))
    del imagelist, time_serie
    #C = nta.CorrelationAnalyzer(ts)
    
    return ts

def get_similarity_timeserie(path, name, condition, time, **kwargs):
    
    TR = 1.
    for arg in kwargs:
        if arg == 'TR':
            TR = np.float(kwargs[arg])
            
    file_list = os.listdir(path)
    
    file_list = [f for f in file_list if f.find(name) != -1 
                                        and f.find('_'+condition) != -1 
                                        and f.find(time) != -1 
                                        ]

    total_data = []
    for f in file_list:
        
        print os.path.join(path, f)
        
        data = np.loadtxt(os.path.join(path, f), delimiter=',')
        data = np.sqrt(data.T)
        
        data_z = zscore(data, axis=1)
        
        total_data.append(data_z)
    
    ts = TimeSeries(np.vstack(total_data), sampling_interval=TR)
    
    return ts
    
def get_condition_timeserie(ts, paradigm, 
                            delete_condition=None,
                            #ts_extraction='mean',
                            paste_runs=False):
    
    '''
    Gets a whole fmri timeserie and returns an object partioned by condition and runs
    
    The output object is a dictionary with conditions as keys with an array runs x roi x timepoints
    
    '''
    
    conditions = paradigm.T[0]
    runs = paradigm.T[1]
    
    if paste_runs == True:
        runs = np.zeros_like(runs)
    
    if delete_condition != None:
        m = conditions != delete_condition
        conditions = conditions[m]
        runs = runs[m] 
    
    cond_list = np.unique(conditions)
    runs_list = np.unique(runs)  
    
    timeserie = dict()
    
    for c in cond_list:
        mask_cond = paradigm.T[0] == c
        
        timeserie[c] = []
        ts_dummy = []
        for r in runs_list:
            mask_run = paradigm.T[1] == r
            
            general_mask = mask_cond * mask_run
            assert general_mask.shape[0] == ts.data.shape[1]
            ts_data = ts.data[:,general_mask]
            print ts_data.shape
            ts_dummy.append(ts_data)
        
        ts_dummy = np.array(ts_dummy)
        timeserie[c] = ts_dummy
            
    return timeserie

#@profile      
def connectivity_analysis(ts_condition, sampling_interval=1.):
    
    matrices = dict()
    for c in ts_condition:
        matrices[c] = []
        for i in range(len(ts_condition[c])):
            
            ts = TimeSeries(ts_condition[c][i], sampling_interval=sampling_interval)
            C = CorrelationAnalyzer(ts)
            
            matrices[c].append(C.corrcoef)
    
    return matrices
    
    
def global_signal_regression(timeserie, regressor):
        
    #Get timeseries data
    Y = timeserie.data.T
        
        
    X = np.expand_dims(regressor, axis=1)
    glm_dist = GeneralLinearModel(X)
    glm_dist.fit(Y)
    beta_dist = glm_dist.get_beta()
        
    r_signal = np.dot(X, beta_dist)
        
    regressed_s = Y - r_signal
    
    return regressed_s



#@profile
def glm(image_ts, regressors):
    
    '''
    image_ts should be a matrix of the form (t x v) where t is the no. of timepoints
    and v is the no. of voxels
    
    regressors should be a matrix of the form (t x n) where t is the no. of timepoints
    and n is the number of regressors
    
    ------------------
    
    beta output is a serie of beta vector of the form (n x v).
    
    '''
    Y = image_ts
        
    if len(regressors.shape) == 1:
        X = np.expand_dims(regressors, axis=1)
    else:
        X = regressors
        

    glm_dist = GeneralLinearModel(X)
    glm_dist.fit(Y)
    
    beta = glm_dist.get_beta()
    
    return beta

#@profile       
def get_bold_signals (image, mask, TR, 
                      normalize=True, 
                      ts_extraction='mean', 
                      filter_par=None, 
                      roi_values=None):
    '''
    Image and mask must be in nibabel format
    '''
    
    mask_data = np.int_(mask.get_data())
    if roi_values == None:
        labels = np.unique(mask_data)[1:]
    else:
        labels = np.int_(roi_values)
    
    final_data = []
    #print labels
    for v in labels[:]:
        #print str(v)
        data = image.get_data()[mask_data == v]
        
        if normalize == True:
            data = zscore(data, axis = 1)
            data[np.isnan(data)] = 0

        if ts_extraction=='mean':
            #assert np.mean(data, axis=0) == data.mean(axis=0)
            data = data.mean(axis=0)
        elif ts_extraction=='pca':
            if data.shape[0] > 0:
                data = PCA(n_components=1).fit_transform(data.T)
                data = np.squeeze(data)
            else:
                data = data.mean(axis=0)
                
        ts = TimeSeries(data, sampling_interval=float(TR))
        
        if filter_par != None:
            
            upperf = filter_par['ub']
            lowerf = filter_par['lb']
            
            F = FilterAnalyzer(ts, ub=upperf, lb=lowerf)
            
            ts = TimeSeries(F.fir.data, sampling_interval=float(TR))
            
            del F
        
        final_data.append(ts.data)

    del data
    del mask_data
    del ts
    return TimeSeries(np.vstack(final_data), sampling_interval=float(TR))

       
def save_matrices(path, results, gsr='gsr', atlas='findlab'):
    
    datetime = get_time()
    analysis = 'connectivity_filtered_first'
    task = 'fmri'
        
    new_dir = '_'.join([datetime,analysis,gsr,atlas,task])
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    parent_dir = os.path.join(path, '0_results', new_dir)
    
    for subj in results.keys():
        
        sub_dir = os.path.join(parent_dir, subj)
        command = 'mkdir '+sub_dir
        
        os.system(command)
        
        for cond in results[subj].keys():
            matrices = results[subj][cond]
            for i in range(len(matrices)):
                
                fname = 'correlation_'+cond+'_run_'+str(i)+'.txt'
                path_fn = os.path.join(sub_dir, fname)
                
                np.savetxt(path_fn, matrices[i], fmt='%.4f')

   
def load_matrices(path, condition):
    """
    path = path of result file
    conditions = analysis label
    """
    subjects = os.listdir(path)
    
    subjects = [s for s in subjects if s.find('configuration') == -1 \
                and s.find('.') == -1]
    
    
    result = []
    
    for c in condition:

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
    
     
def z_fisher(r):
    
    F = 0.5*np.log((1+r)/(1-r))
    
    return F

