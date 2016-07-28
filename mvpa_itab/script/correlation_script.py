# pylint: disable=maybe-no-member, method-hidden, undefined-variable
from mvpa_itab.io.base import load_wu_file_list
from mvpa_itab.connectivity import *
from nitime.timeseries import TimeSeries
from memory_profiler import profile


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
    
    coher_dist = []
    coher_bold = []
    
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
        C_pre_d = CorrelationAnalyzer(dist_ts_pre_deconv)
        
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

    correlation_list = []
    connectivity_list = []
    
    coher_dist = []
    coher_bold = []
    
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
        
        dist_ts_pre = TimeSeries(dist_ts_file.T, sampling_interval=np.float(conf['tr']))
        
        assert dist_ts_pre.data.shape[0] == bold_ts_pre.data.shape[0]
        
        C_dist_pre = CorrelationAnalyzer(dist_ts_pre)
        C_bold_pre = CorrelationAnalyzer(bold_ts_pre)
        
        Ch_bold_pre = CoherenceAnalyzer(bold_ts_pre)
        Ch_dist_pre = CoherenceAnalyzer(dist_ts_pre)
        
        #################### Post #########################
        
        list_post = [f for f in imglist if f.get_filename().split('/')[-2] == 'task']
        
        length_post = len(list_post)
        
        conf['runs'] = np.int(length_pre)
        
        bold_ts_post = get_bold_timeserie(list_post, path_roi, roi_names, ts_param, **conf)
        #dist_ts_post = get_similarity_timeserie(path_dist, name, condition, 'RestPost', **conf)
    
        fname = '_'.join(['similarity','regressed',name, condition, 'RestPost','.txt'])
        dist_ts_file = np.loadtxt(os.path.join(path_dist,fname), delimiter=',')
        
        dist_ts_post = TimeSeries(dist_ts_file.T, sampling_interval=np.float(conf['tr']))
        
        assert dist_ts_post.data.shape[0] == bold_ts_post.data.shape[0]
        
        C_dist_post = CorrelationAnalyzer(dist_ts_post)
        C_bold_post = CorrelationAnalyzer(bold_ts_post)
        
        Ch_bold_post = CoherenceAnalyzer(bold_ts_post)
        Ch_dist_post = CoherenceAnalyzer(dist_ts_post)
        
        
        correlation_list.append([C_dist_pre, C_dist_post])
        connectivity_list.append([C_bold_pre, C_bold_post])
        
        coher_dist.append([Ch_dist_pre, Ch_dist_post])
        coher_bold.append([Ch_bold_pre, Ch_bold_post])
                
        del list_pre, list_post, imglist
    
    
##################################################################

    path = '/media/DATA/fmri/monks'
    
    '''
    brain_mask  = ni.load('/media/DATA/fmri/templates_MNI_3mm/MNI152_T1_3mm_brain_mask.nii.gz')
    wm_mask = ni.load('/media/DATA/fmri/templates_MNI_3mm/wm_MNI_3mm.nii.gz')
    ventricles_mask = ni.load('/media/DATA/fmri/templates_MNI_3mm/ventricles_MNI_3mm.nii.gz')
    '''    
    
    for sub in dlist:
        
        print 'Analyzing '+sub
        sub_path = os.path.join(path, sub, 'fmri')
        
        brain_mask = ni.load(os.path.join(path, sub, 'fmri', 'bold_orient_mask_mask.nii.gz'))
        wm_mask = ni.load(os.path.join(path, sub, 'fmri', 'mprage_orient_brain_seg_wm_333.nii.gz'))
        ventricles_mask = ni.load(os.path.join(path, sub, 'fmri', 'mprage_orient_brain_seg_csf_333.nii.gz'))
        fmri_name = os.path.join(path, sub, 'fmri', 'bold_orient.nii.gz')
        
        fmri = ni.load(fmri_name)
        
        fmri_ts = get_bold_signals(fmri, brain_mask, 4., ts_extraction='none')
        
        global_signal = get_bold_signals(fmri, brain_mask, 4.)
        ventricles_signal = get_bold_signals(fmri, ventricles_mask, 4.)
        wm_signal = get_bold_signals(fmri, wm_mask, 4.)
    
        regressors = np.vstack((
                                #global_signal.data, 
                                ventricles_signal.data, 
                                wm_signal.data
                                ))
        
        del wm_signal, ventricles_signal, global_signal
        print 'Regressing out Global, Ventricles and White Matter signals...'
        beta = glm(fmri_ts.data.T, regressors.T)
        
        print 'Getting residuals...'
        residuals = fmri_ts.data.T - np.dot(regressors.T, beta)
        
        del regressors, fmri_ts, beta
        print 'Filtering data...'
        ts_residual = TimeSeries(residuals.T, sampling_interval=4.)
        F = FilterAnalyzer(ts_residual, ub=0.08, lb=0.009)
        
        residual_4d = np.zeros_like(fmri.get_data())
        residual_4d [brain_mask.get_data() > 0] = F.fir.data
        residual_4d[np.isnan(residual_4d)] = 0
        
        ni.save(ni.Nifti1Image(residual_4d, fmri.get_affine()), 
                os.path.join(path, sub, 'fmri', 'residual_filtered_no_gsr.nii.gz'))
        
        del residual_4d, F, fmri, residuals, brain_mask, wm_mask, ventricles_mask
        gc.collect()
        
    
##########################################################################

'''
roi_list = np.loadtxt(
                          '/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                          dtype=np.str,
                          delimiter=','
                          )
'''
roi_list = np.loadtxt(
                          '/media/robbis/DATA/fmri/templates_AAL/atlas90.cod',
                          dtype=np.str,
                          delimiter='='
                          )

path = '/media/robbis/DATA/fmri/monks'
#brain_mask  = ni.load('/media/DATA/fmri/templates_MNI_3mm/MNI152_T1_3mm_brain_mask.nii.gz')
paradigm_path = '/media/robbis/DATA/fmri/monks/monks_attrib_pymvpa.txt'
paradigm = np.loadtxt(
                          paradigm_path,
                          dtype=np.str,
                          delimiter=' '
                          )
    
combinations = [#['no_gsr', True],
                #['no_gsr', False],
                #['_gsr',True],
                #['_gsr', False]
                ['raw',False],
                ]

for it in combinations:
    results = dict()
    for subj in dlist[:]:
        print ' ****** '+subj+' ****** '
        fmri_name = os.path.join(path, subj, 'fmri', 
                                 #'residual_filtered_first'+it[0]+'.nii.gz')
                                 #'residual_filtered_'+it[0]+'.nii.gz')
                                 'bold_orient.nii.gz')
        fmri = ni.load(fmri_name)
        
        # = ni.load(os.path.join(path,subj,'fmri','bold_orient_mask_mask.nii.gz'))
        
        data = []
        
        mask = ni.load(os.path.join(path, subj, 'fmri', 'atlas90_brain_seg_gm_333.nii.gz'))
        
        # With findlab rois we loop across networks
        for roi in np.unique(np.int_(roi_list.T[0])):
        #for roi in np.unique((roi_list.T[-2])):   
            
            ### Uncomment for find lab rois analysis ###
            #roi_name = roi.split('_')[0]
            
            #roi_name = roi.split('#')[0]
            #mask_name = "%s_separated_anat_333.nii.gz" % (roi_name)
            #mask_path = os.path.join(path, subj, 'fmri', mask_name)
                                     
            #print mask_path 
            #mask = ni.load(mask_path)
            
            
            #roi_val = roi_list[roi_list.T[-2] == roi].T[-1]
            ####
            
            roi_val = [roi]
            
            mask_ts = get_bold_signals(fmri, mask, 4., 
                                       roi_values=roi_val,
                                       ts_extraction='mean')
            
            data.append(mask_ts.data)
        
        data = np.vstack(data)
        
        print data.shape
        
        ts = TimeSeries(data, sampling_interval=4.)
        
        ts_condition = get_condition_timeserie(ts, paradigm, 
                                               delete_condition='Rest',
                                               #ts_extraction='pca',
                                               #delete_condition=None,
                                               paste_runs=it[1])
        #save timeseries
        
        matrices = connectivity_analysis(ts_condition, sampling_interval=ts.sampling_interval)
        
        results[subj] = matrices
        #del ts_condition, ts, fmri, data, mask_ts
    save_matrices(path, results, gsr='filtered_after_each_run_'+it[0], atlas='atlas90')
    
################################################


    
