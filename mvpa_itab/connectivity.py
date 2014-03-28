import nitime.analysis as nta
import nitime.fmri.io as io

from nipy.modalities.fmri.glm import GeneralLinearModel

from nitime.timeseries import TimeSeries
from nitime.analysis.spectral import FilterAnalyzer
from nitime.analysis.correlation import CorrelationAnalyzer

from scipy.signal.windows import boxcar
from scipy.signal.signaltools import convolve, detrend
from scipy.stats.mstats import zscore

from lib_io import load_wu_fmri_data, read_configuration

import matplotlib.pyplot as pl

import os
import nibabel as ni
import numpy as np

#@profile
def analyze_connectivity(imagelist, path_roi, roi_names, ts_param, **kwargs):
    
    TR = 1.
    for arg in kwargs:
        if arg == 'TR':
            TR = np.float(kwargs[arg])
    
    roi_list = os.listdir(path_roi)
    roi_list = [r for r in roi_list if r.find('.hdr') != -1 or r.find('nii.gz') != -1]
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
                    data = np.loadtxt(os.path.join(path_file, '_'.join([s,'distance','txt',cond,time,'.txt'])))
                    
                    data_list.append(data)
                    roi_list.append(roi)
                
                data_a = np.vstack(data_list).T
                roi_a = np.array(roi_list)
                
                path_save = os.path.join(path, '0_results', 'connectivity', 'similarity')
                
                np.savetxt(os.path.join(path_save, '_'.join(['similarity',cond,time,s,'.txt'])), data_a, 
                           fmt='%4.4f', delimiter=',')
    np.savetxt(os.path.join(path_save, 'roi_labels.txt'), roi_a, fmt='%s')
    
    '''
    for f in file_list[:11]:
        for u, l in zip(ub, lb):
        
            data = np.loadtxt(os.path.join(path_save, f),delimiter=',')
            data = np.sqrt(data.T)
            data_z = zscore(data, axis=1)
        
            ts = TimeSeries(data_z, sampling_interval=float(TR))
            F = FilterAnalyzer(ts, ub=u, lb=l)
            
            f_ts = F.fir.data
            
            A = CorrelationAnalyzer(F.fir)
            
            #np.savetxt(os.path.join(path_save, 'correlation', f[:f.find('.txt')]+'_corr.txt'), A.corrcoef, fmt='%.4f')
            pl.figure()
            pl.plot(f_ts.T)
            pl.plot(-data_z.T)
            pl.savefig(os.path.join(path_save, 'correlation', f[:f.find('.txt')]+'_ts_'+str(l)+'_'+str(u)+'_.png'))
            
            pl.figure()
            pl.imshow(A.corrcoef, interpolation='nearest')
            pl.xticks(np.arange(len(roi_a)), roi_a, rotation='vertical')
            pl.yticks(np.arange(len(roi_a)), roi_a)
            pl.colorbar()
            pl.savefig(os.path.join(path_save, 'correlation', f[:f.find('.txt')]+'_corr'+str(l)+'_'+str(u)+'.png'))
            
        corr_list.append(A.corrcoef)
    '''
                        
    
    return 

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
    
    TR = 1.
    for arg in kwargs:
        if arg == 'TR':
            TR = np.float(kwargs[arg])
            
    roi_list = os.listdir(path_roi)
    roi_list = [r for r in roi_list if r.find('.hdr') != -1 or r.find('nii.gz') != -1]
    
    print 'Length of image list is '+str(len(imagelist))
    volume_shape = imagelist[0].get_data().shape[:-1]
    n_shape = list(volume_shape)
    n_shape.append(-1)
    coords = list(np.ndindex(volume_shape))
    coords = np.array(coords).reshape(n_shape)
    n_runs = len(imagelist)
    
    data_arr = []
    for roi in roi_names.T[0]:
        #print roi
        r_mask = ni.load(os.path.join(path_roi, roi+'nii.gz'))
        mask = r_mask.get_data().squeeze()
        roi_filt = roi_names[roi_names.T[0] == roi]
        
        for label in np.unique(mask)[1:]:
            
            roi_m = np.int_(roi_filt.T[2]) == label
            if roi_m.any():
                print 'Loading voxels from '+roi_filt[roi_m].T[0][0]
                time_serie = io.time_series_from_file([f.get_filename() for f in imagelist], \
                                                      coords[mask==label].T, \
                                                      TR=float(TR), \
                                                      normalize=None, \
                                                      average=None, \
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
                    
                
                if ts_param['average'] == True:
                    ts_new = np.mean(time_serie.data, axis=0)
                    
                
                data_arr.append(ts_new)
        del r_mask
    
    data = np.array(data_arr)
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
    

        
def connectivity(ts):
    
    C = nta.CorrelationAnalyzer(ts)
    
    return C


def global_signal_regression(timeserie, regressor):
        
        #Get timeseries data
        Y = timeserie.data.T
        
        
        X = np.expand_dims(regressor, axis=1)
        glm_dist = GeneralLinearModel(X)
        glm_dist.fit(Y)
        beta_dist = glm_dist.get_beta()
        
        r_signal = np.dot(X, beta_dist)
        
        regressed_s = Y - r_signal
    


def plot_cross_correlation(xcorr, t_start, t_end, labels):


    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    dim = len(labels)
    
    
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.5, dim-0.5), ylim=(dim-0.5, -0.5))
    
    
    #im = ax.imshow(xcorr.at(t_start), interpolation='nearest', vmin=-1, vmax=1)
    im = ax.imshow(np.eye(dim), interpolation='nearest', vmin=-1, vmax=1)
    title = ax.set_title('')
    xt = ax.set_xticks(np.arange(dim))
    xl = ax.set_xticklabels(labels, rotation='vertical')
    yt = ax.set_yticks(np.arange(dim))
    yl = ax.set_yticklabels(labels)
    fig.colorbar(im)

    l_time = np.arange(-50, 50, 1)
    mask = (l_time >= t_start) * (l_time<=t_end)
    
    def init():
        im.set_array(np.eye(dim))
        title.set_text('Cross-correlation at time lag of '+str(t_start)+' TR.')
        plt.draw()
        return im, title
        
    
    
    def animate(i):
        global l_time        
        j = np.int(np.rint(i/10))
        print l_time[mask][j]
        #im.set_array(xcorr.at(l_time[j]))
        im.set_array(x[mask][j])
        title.set_text('Cross-correlation at time lag of '+str(l_time[mask][j])+' TR.')
        plt.draw()
        return im, title

    ani = animation.FuncAnimation(fig, animate, 
                                  init_func=init, 
                                  frames=10*(t_end-t_start), 
                                  interval=20,
                                  repeat=False, 
                                  blit=True)
    plt.show()
    #ani.save('/home/robbis/xcorrelation_.mp4')

    
################### Script ##########################
if __name__ == '__main__':
    path = ''
    subjects = []
    
    task = 'rest'
    conf = read_configuration(path, 'learning.conf', task)
    
    path_data = '/media/DATA/fmri/learning/'
    path_roi = '/media/DATA/fmri/learning/1_single_ROIs/fcMRI_ROI/single_ROI/'
    path_dist = '/media/DATA/fmri/learning/0_results/connectivity/similarity/regressed_distance'
    path_dist_save = '/media/DATA/fmri/learning/0_results/connectivity/similarity/deconvolved_distances'
    
    roi_names = np.loadtxt('/media/DATA/fmri/learning/roi_labels', dtype=np.str)
    
    condition = 'trained'
    
    ts_param = dict()
    ts_param['filter'] = {'lb': 0.008, 'ub':0.09, 'method':'fir'}
    ts_param['average'] = True
    ts_param['normalize'] = 'zscore'
    
    correlation_list = []
    connectivity_list = []
    for name in subjects:
        
        imglist = load_wu_fmri_data(path_data, name, task, **conf)
        
        ################## Pre ##########################
        
        list_pre = [f for f in imglist if f.get_filename().split('/')[-2] == 'rest']
        
        length_pre = len(list_pre)
        
        print length_pre
        conf['runs'] = np.int(length_pre)
        
        bold_ts_pre = get_bold_timeserie(list_pre, path_roi, roi_names, ts_param, **conf)
        dist_ts_pre = get_similarity_timeserie(path_dist, name, condition, 'RestPre', **conf)
        
        '''
        bold_ts_pre_conv = bold_convolution(bold_ts_pre, 7, win_func=boxcar)
        
        dist_ts_pre_deconv = remove_bold_effect(bold_ts_pre_conv, dist_ts_pre, ts_param, **conf)
        
        fname = os.path.join(path_dist_save, 'ts_deconv_'+name+'_'+condition+'_RestPre.txt')
        print dist_ts_pre_deconv.shape
        
        np.savetxt(fname, dist_ts_pre_deconv.data, fmt='%.6f', delimiter=',')
        '''
        
        C_pre_nd = CorrelationAnalyzer(dist_ts_pre)
        #C_pre_d = CorrelationAnalyzer(dist_ts_pre_deconv)
        
        fname_c = os.path.join(path_dist_save, 'corr_deconv_'+name+'_'+condition+'_RestPre.txt')
        np.savetxt(fname_c, C_pre_d.corrcoef, fmt='%.6f', delimiter=',')
        
        fname_nc = os.path.join(path_dist_save, 'corr_true_'+name+'_'+condition+'_RestPre.txt')
        np.savetxt(fname_nc, C_pre_nd.corrcoef, fmt='%.6f', delimiter=',')
        
        '''
        f = pl.figure()
        ax1 = f.add_subplot(121)
        m1 = ax1.imshow(C_pre_nd.corrcoef, interpolation='nearest')
        f.colorbar(m1)
        ax2 = f.add_subplot(122)
        m2 = ax2.imshow(C_pre_d.corrcoef, interpolation='nearest')
        f.colorbar(m2)
        '''
        
        #################### Post #########################
        
        list_post = [f for f in imglist if f.get_filename().split('/')[-2] == 'task']
        
        length_post = len(list_post)
        
        conf['runs'] = np.int(length_pre)
        
        bold_ts_post = get_bold_timeserie(list_post, path_roi, roi_names, ts_param, **conf)
        dist_ts_post = get_similarity_timeserie(path_dist, name, condition, 'RestPost', **conf)
        
        bold_ts_post_conv = bold_convolution(bold_ts_post, 7, win_func=boxcar)
        
        dist_ts_post_deconv = remove_bold_effect(bold_ts_post_conv, dist_ts_post, ts_param, **conf)
        print dist_ts_post_deconv.shape
        
        fname = os.path.join(path_dist_save, 'ts_deconv_'+name+'_'+condition+'_RestPost.txt')
        np.savetxt(fname, dist_ts_post_deconv.data, fmt='%.6f', delimiter=',')
    
        C_post_nd = CorrelationAnalyzer(dist_ts_post)
        C_post_d = CorrelationAnalyzer(dist_ts_post_deconv)
        
        fname_c = os.path.join(path_dist_save, 'corr_deconv_'+name+'_'+condition+'_RestPost.txt')
        np.savetxt(fname_c, C_post_d.corrcoef, fmt='%.6f', delimiter=',')
        
        fname_nc = os.path.join(path_dist_save, 'corr_true_'+name+'_'+condition+'_RestPost.txt')
        np.savetxt(fname_nc, C_post_nd.corrcoef, fmt='%.6f', delimiter=',')
        
        correlation_list.append([C_pre_d.corrcoef, C_post_d.corrcoef, C_pre_nd.corrcoef, C_post_nd.corrcoef])
        connectivity_list.append([bold_ts_post, bold_ts_pre])
        
        '''
        f = pl.figure()
        ax1 = f.add_subplot(121)
        m1 = ax1.imshow(C_post_nd.corrcoef, interpolation='nearest')
        f.colorbar(m1)
        ax2 = f.add_subplot(122)
        m2 = ax2.imshow(C_post_d.corrcoef, interpolation='nearest')
        f.colorbar(m2)
        '''
        
    for name in subjects:
    
        fname = os.path.join(path_dist_save, 'ts_deconv_'+name+'_'+condition+'_RestPost.txt')
    
        dist_ts_post_deconv = np.loadtxt(fname, delimiter=',')
        dist_ts_post = get_similarity_timeserie(path_dist, name, condition, 'RestPost', **conf)
    
        f = pl.figure()
        ax1 = f.add_subplot(211)
    
    ################ Global Signal Regression #########################
    
    res = []
    
    for rest in ['RestPre', 'RestPost']:
        
        s_res = []
        
        for name in subjects:
        ################ Post #############
        
            dist_ts = get_similarity_timeserie(path_dist, name, condition, rest, **conf)
        
            Y = dist_ts.data.T
            regressor = np.mean(Y, axis=1)
        
            X = np.expand_dims(regressor, axis=1)
            glm_dist = GeneralLinearModel(X)
            glm_dist.fit(Y)
            beta_dist = glm_dist.get_beta()
        
            r_signal = np.dot(X, beta_dist)
        
            regressed_s = Y - r_signal
            
            save_fn = os.path.join(path_dist,'similarity_regressed_'+name+'_'+condition+'_'+rest+'_.txt')
            #np.savetxt(save_fn, regressed_s, fmt='%.4f', delimiter=',')
            
            r_ts = TimeSeries(regressed_s.T, sampling_interval=dist_ts.sampling_interval)
            
            C = CorrelationAnalyzer(r_ts)
            s_res.append(np.arctanh(C.corrcoef))
        
        res.append(s_res)
            
######################## Cross Correlation ###################################


    for name in subjects:
        
        imglist = load_wu_fmri_data(path_data, name, task, **conf)
        
        ################## Pre ##########################
        
        list_pre = [f for f in imglist if f.get_filename().split('/')[-2] == 'rest']
        
        length_pre = len(list_pre)
        
        print length_pre
        conf['runs'] = np.int(length_pre)
        
        bold_ts_pre = get_bold_timeserie(list_pre, path_roi, roi_names, ts_param, **conf)
        #dist_ts_pre = get_similarity_timeserie(path_dist, name, condition, 'RestPre', **conf)
        
        fname = '_'.join(['similarity','regressed',name, condition, 'RestPre','.txt'])
        dist_ts_file = np.loadtxt(os.path.join(path_dist,fname), delimiter=',')
        
        dist_ts_pre = TimeSeries(dist_ts_file.T, sampling_interval=bold_ts_pre.sampling_interval)
        
        assert dist_ts_pre.data.shape[0] == bold_ts_pre.data.shape[0]
        
        C_dist_pre = CorrelationAnalyzer(dist_ts_pre)
        C_bold_pre = CorrelationAnalyzer(bold_ts_pre)
        
        
        
        #################### Post #########################
        
        list_post = [f for f in imglist if f.get_filename().split('/')[-2] == 'task']
        
        length_post = len(list_post)
        
        conf['runs'] = np.int(length_pre)
        
        bold_ts_post = get_bold_timeserie(list_post, path_roi, roi_names, ts_param, **conf)
        #dist_ts_post = get_similarity_timeserie(path_dist, name, condition, 'RestPost', **conf)
    
        fname = '_'.join(['similarity','regressed',name, condition, 'RestPost','.txt'])
        dist_ts_file = np.loadtxt(os.path.join(path_dist,fname), delimiter=',')
        
        dist_ts_post = TimeSeries(dist_ts_file.T, sampling_interval=bold_ts_pre.sampling_interval)
        
        assert dist_ts_post.data.shape[0] == bold_ts_post.data.shape[0]
        
        C_dist_post = CorrelationAnalyzer(dist_ts_post)
        C_bold_post = CorrelationAnalyzer(bold_ts_post)
        
        correlation_list.append([C_dist_pre.xcorr_norm, C_dist_post.xcorr_norm])
        connectivity_list.append([C_bold_pre.xcorr_norm, C_bold_pre.xcorr_norm])
        
        del list_pre, list_post, imglist
    