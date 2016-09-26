import os
import numpy as np
import matplotlib.pyplot as pl
from mvpa_itab.conn.io import ConnectivityLoader
from mvpa_itab.conn.operations import copy_matrix
from mvpa_itab.conn.utils import get_atlas_info
from mvpa_itab.conn.plot import *
import nibabel as ni
from scipy.stats.stats import zscore



def save_results(path, results):
    
    fields_ = ['error', 'features', 'weights', 'subjects']
    
    for result in results:
    
        conf_ = result[0]
        real_res = result[1][0][0]
        perm_res = result[1][0][1]
        
                    
        array_save = dict()
        perm_save = dict()
        
        for i, _ in enumerate(real_res):
            array_save[fields_[i]] = np.array(real_res[i])
            perm_save[fields_[i]] = np.array([np.vstack([f for f in s]) for s in perm_res[:,i,:]])
            
        
        p_name = build_name(path, conf_, field='permutation', perm=1000, cv=50)
        v_name = build_name(path, conf_, field='values', perm=1000, cv=50)
        
        np.savez_compressed(p_name, perm_save)
        np.savez_compressed(v_name, array_save)
            
    
    
    
            
def build_name(path, conf, field='values', **kwargs):
    
    fname = os.path.join(path, conf['directory'])
    alg = conf['learner'].__str__()[:3]
    kernel = conf['learner'].kernel
    med_ = conf['conditions']
    
    return os.path.join(fname,
                        "%s_%s_%s_%s.npz" %(med_, 
                                              #alg, 
                                              #kernel, 
                                              field, 
                                              str(kwargs['perm']), 
                                              str(kwargs['cv'])) )       
        



def analyze_results(directory, 
                    conditions, 
                    n_permutations=1000.):
    
    
    """Write the results of the regression analysis

    Parameters
    ----------
    directory : string or list of strings
        Path or list of paths where put results.
    
    condition : string or list of strings
        Conditions to be analyzed.


    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.

    """
    
    res_path = '/media/robbis/DATA/fmri/monks/0_results/'
    subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)

    path = '/media/robbis/DATA/fmri/monks/'
    roi_list = []
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                          delimiter=',',
                          dtype=np.str)
    
    if isinstance(directory, str):
        directory = [directory]
        
    if isinstance(conditions, str):
        conditions = [conditions]
        
    
    for dir_ in directory:
        for cond_ in conditions:
            
            fname_ = os.path.join(res_path, dir_, cond_+'_values_1000_50.npz')
            
            results_ = np.load(fname_)
            values_ = results_['arr_0'].tolist()
            errors_ = values_['error']      #values_['errors_']
            sets_ = values_['features']     #values_['sets_']
            weights_ = values_['weights']   #values_['weights_']
            samples_ = values_['subjects']  #values_['samples_']
            
            fname_ = os.path.join(res_path, dir_, cond_+'_permutation_1000_50.npz')
            
            results_ = np.load(fname_)
            values_p = results_['arr_0'].tolist()
            errors_p = values_p['error']        #values_p['errors_p']
            sets_p = values_p['features']       #values_p['sets_p']
            weights_p = values_p['weights']     #values_p['weights_p']
            samples_p = values_p['subjects']    #values_p['samples_p']
            
            errors_p = np.nanmean(errors_p, axis=1)
                        
            print('-----------'+dir_+'-------------')
            print(cond_)
            print ('MSE = '+str(errors_[:,0].mean())+' -- p '+ \
                str(np.count_nonzero(errors_p[:,0] < errors_[:,0].mean())/n_permutations))
            print('COR = '+str(np.nanmean(errors_[:,1]))+' -- p '+ \
                str(np.count_nonzero(errors_p[:,1] > np.nanmean(errors_[:,1]))/n_permutations))
                
            directory_ = dir_
            learner_ = "SVR_C_1" 
        
            prename = "%s_%s" %(cond_, learner_)
            
            ######## Get matrix infos ###############
            
            conn_test = ConnectivityLoader(res_path, 
                                         subjects, 
                                         directory_, 
                                         roi_list)
            
            # Get nan mask to correctly fill matrix
            nan_mask = conn_test.get_results(['Samatha', 'Vipassana'])
            # Transform matrix into float of ones
            mask_ = np.float_(~np.bool_(nan_mask))
            # Get the upper part of the matrix
            mask_ = np.triu(mask_, k=1)
            mask_indices = np.nonzero(mask_)
            n_bins = np.count_nonzero(mask_)
            
            
            ###### Plot of distributions of errors and permutations #########
            #errors_p = np.nanmean(errors_p, axis=1)
            
            fig_ = pl.figure()
            bpp = pl.boxplot(errors_p, showfliers=False, showmeans=True, patch_artist=True)
            bpv = pl.boxplot(errors_, showfliers=False, showmeans=True, patch_artist=True)
            fname = "%s_perm_1000_boxplot.png" %(prename)
           
            
            for box_, boxp_ in zip(bpv['boxes'], bpp['boxes']):
                box_.set_facecolor('lightgreen')
                boxp_.set_facecolor('lightslategrey')
              
              
            pl.xticks(np.array([1,2]), ['MSE', 'COR'])
            
            pl.savefig(os.path.join(res_path, directory_, fname))
            pl.close()
            
            n_permutations = np.float(errors_p[:,0].shape[0])
            
            
            ##### Plot of connection distributions ########
            
            pl.figure()
            h_values_p, _ = np.histogram(sets_p.flatten(), bins=np.arange(0, n_bins+1))
            #pl.plot(zscore(h_values_p))
            
            pl.hist(zscore(h_values_p), bins=25)
            
            fname = "%s_features_set_dist.png" %(prename)
            pl.savefig(os.path.join(res_path, directory_, fname))
            
            pl.figure()
            h_values_, _ = np.histogram(sets_.flatten(), bins=np.arange(0, n_bins+1))
            pl.plot(zscore(h_values_))
                
            
            fname = "%s_features_set_cross_validation.png" %(prename)
            pl.savefig(os.path.join(res_path, directory_, fname))
            
            pl.close('all')
            
            
            ######## Plot connectivity stuff ###########
            
            weights_ = weights_.squeeze()
            filling_vector = np.zeros(np.count_nonzero(mask_))
            counting_vector = np.zeros(np.count_nonzero(mask_))
            
            for s, w in zip(sets_, weights_):
                filling_vector[s] += zscore(w)
                counting_vector[s] += 1
            
            # Calculate the average weights and then zscore
            avg_weigths = np.nan_to_num(filling_vector/counting_vector)
            
            mask_[mask_indices] = avg_weigths
            
            matrix_ = np.nan_to_num(copy_matrix(mask_, diagonal_filler=0))
        
            names_lr, colors_lr, index_, coords, _ = get_atlas_info(dir_)
            
            '''
            matrix_[matrix_ == 0] = np.nan
            matrix_[np.abs(matrix_) < 1] = np.nan
            '''
            size_w = np.zeros_like(matrix_)
            size_w[mask_indices] = np.abs(avg_weigths)
            size_w = np.nan_to_num(copy_matrix(size_w, diagonal_filler=0))
            size_w = np.sum(size_w, axis=0)
            
            f, _ = plot_connectivity_circle_edited(matrix_[index_][:,index_], 
                                            names_lr[index_], 
                                            node_colors=colors_lr[index_],
                                            node_size=2*size_w[index_]**2,
                                            con_thresh = 1.4,
                                            title=cond_,
                                            node_angles=circular_layout(names_lr, 
                                                                        list(names_lr),
                                                                        ),
                                            fontsize_title=19,
                                            fontsize_names=13,
                                            fontsize_colorbar=13,
                                            colorbar_size=0.3,
                                            colormap='bwr',
                                            #colormap=cm_,
                                            vmin=-3.,
                                            vmax=3.,
                                            fig=pl.figure(figsize=(16,16))
                                            )
            
            
            fname = "%s_features_weight.png" %(prename)
            f.savefig(os.path.join(res_path, directory_, fname),
                      facecolor='black',
                      dpi=150)
            for d_ in ['x', 'y', 'z']:
                fname = "%s_connectome_feature_weight_%s.png" %(prename, d_)
                fname = os.path.join(res_path, directory_, fname)
                plot_connectome(matrix_, 
                                coords, 
                                colors_lr, 
                                2*size_w**2,
                                1.4,
                                fname,
                                #cmap=pl.cm.bwr,
                                title=None,
                                display_=d_,
                                #max_=3.,
                                #min_=3. 
                                )
            fname = "%s_connections_list_feature_weights.txt" %(prename)
            fname = os.path.join(res_path, directory_, fname)
            #print_connections(matrix_, names_lr, fname)
            
            #########
            mask_ = np.float_(~np.bool_(nan_mask))
            mask_ = np.triu(mask_, k=1)
            mask_indices = np.nonzero(mask_)
            mask_[mask_indices] = h_values_
            matrix_ = np.nan_to_num(copy_matrix(mask_, diagonal_filler=0))
            
            size_ = np.zeros_like(matrix_)
            size_[mask_indices] = counting_vector
            size_ = np.nan_to_num(copy_matrix(size_, diagonal_filler=0))
            size_ = np.sum(size_, axis=0)
            
            f, _ = plot_connectivity_circle_edited(matrix_[index_][:,index_], 
                                            names_lr[index_], 
                                            node_colors=colors_lr[index_],
                                            node_size=size_[index_]*5,
                                            con_thresh = 15.,
                                            title=cond_,
                                            node_angles=circular_layout(names_lr, 
                                                                        list(names_lr),
                                                                        ),
                                            fontsize_title=19,
                                            fontsize_names=13,
                                            fontsize_colorbar=13,
                                            colorbar_size=0.3,
                                            #colormap='bwr',
                                            #colormap='terrain',
                                            #vmin=40,
                                            fig=pl.figure(figsize=(16,16))
                                            )
            
            fname = "%s_features_choices.png" %(prename)
            f.savefig(os.path.join(res_path, directory_, fname),
                      facecolor='black',
                      dpi=150)
            
            for d_ in ['x', 'y', 'z']:
                fname = "%s_connectome_feature_choices_%s.png" %(prename, d_)
                fname = os.path.join(res_path, directory_, fname)
                
                plot_connectome(matrix_, 
                                coords, 
                                colors_lr, 
                                4.*size_,
                                15.,
                                fname,
                                title=None,
                                max_=50.,
                                min_=0.,
                                display_=d_
                                )
                
            fname = "%s_connections_list_feature_choices.txt" %(prename)
            fname = os.path.join(res_path, directory_, fname)
            #print_connections(matrix_, names_lr,fname)
            
            pl.close('all')
            

