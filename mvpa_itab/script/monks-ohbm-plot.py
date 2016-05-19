from mvpa_itab.conn.utils import ConnectivityTest
from mvpa_itab.conn.io import copy_matrix
from mvpa_itab.conn.plot import *
import nibabel as ni
from scipy.stats.stats import zscore



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
        """
        dict_ = {'Auditory':'#89CC74', 
                 'Basal_Ganglia':'#7A84CC', 
                 'LECN':'#FF1800',
                 'Language':'#BF2B54', 
                 'Precuneus':'#390996',
                 'RECN':'#FF230B', 
                 'Sensorimotor':'#4D0DC8', 
                 'Visuospatial':'#DBBF00', 
                 'anterior_Salience':'#37AEC4',
                 'dorsal_DMN':'#9AF30B', 
                 'high_Visual':'#FF8821', 
                 'post_Salience':'#0289A2', 
                 'prim_Visual':'#FF7600',
                 'ventral_DMN':'#92ED00'
                 }"""
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



def write_results(directory_list, conditions, n_permutations=1000.):
    
    res_path = '/media/robbis/DATA/fmri/monks/0_results/'
    roi_list = []
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                          delimiter=',',
                          dtype=np.str)
    for dir_ in directory_list:
        for cond_ in conditions:
            
            fname_ = os.path.join(res_path, dir_, cond_+'_values_1000_cv_50.npz')
            
            results_ = np.load(fname_)
            values_ = results_['arr_0'].tolist()
            errors_ = values_['errors_']
            sets_ = values_['sets_']
            weights_ = values_['weights_']
            samples_ = values_['samples_']
            
            fname_ = os.path.join(res_path, dir_, cond_+'_permutation_1000_cv_50.npz')
            
            results_ = np.load(fname_)
            values_p = results_['arr_0'].tolist()
            errors_p = values_p['errors_p']
            sets_p = values_p['sets_p']
            weights_p = values_p['weights_p']
            samples_p = values_p['samples_p']
            
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
            conn_test = ConnectivityTest(path, 
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
        
            names_lr, colors_lr, index_, coords = get_plot_stuff(directory_)
            
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
            #f.savefig(os.path.join(res_path, directory_, fname),
            #          facecolor='black',
            #          dpi=150)
            for d_ in ['x', 'y', 'z']:
                fname = "%s_connectome_feature_weight_%s.png" %(prename, d_)
                fname = os.path.join(res_path, directory_, fname)
                plot_connectome(matrix_, 
                                coords, 
                                colors_lr, 
                                2*size_w**2,
                                index_,
                                1.4,
                                fname,
                                cmap=pl.cm.bwr,
                                title=None,
                                display_=d_,
                                #max_=3.,
                                #min_=3. 
                                )
            fname = "%s_connections_list_feature_weights.txt" %(prename)
            fname = os.path.join(res_path, directory_, fname)
            print_connections(matrix_, names_lr, fname)
            
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
            #f.savefig(os.path.join(res_path, directory_, fname),
            #          facecolor='black',
            #          dpi=150)
            
            for d_ in ['x', 'y', 'z']:
                fname = "%s_connectome_feature_choices_%s.png" %(prename, d_)
                fname = os.path.join(res_path, directory_, fname)
                
                plot_connectome(matrix_, 
                                coords, 
                                colors_lr, 
                                4.*size_,
                                index_,
                                15.,
                                fname,
                                title=None,
                                max_=50.,
                                min_=0.,
                                display_=d_
                                )
                
            fname = "%s_connections_list_feature_choices.txt" %(prename)
            fname = os.path.join(res_path, directory_, fname)
            print_connections(matrix_, names_lr,fname)
            
            pl.close('all')   
    
###################################################

def plot_connectome(matrix_, 
                    coords, 
                    colors, 
                    size, 
                    order_, 
                    threshold, 
                    fname, cmap=pl.cm.hot, title='', max_=None, min_=None, display_='ortho'):
    
    from nilearn import plotting
    plotting.plot_connectome(adjacency_matrix=matrix_[order_][:,order_], 
                             node_coords=coords[order_], 
                             node_color=colors[order_].tolist(), 
                             node_size=1.5*size[order_], 
                             edge_cmap=cmap, 
                             edge_vmin=min_, 
                             edge_vmax=max_, 
                             edge_threshold=threshold, 
                             output_file=fname, 
                             display_mode=display_, 
                             figure=pl.figure(figsize=(16*1.2,9*1.2)),# facecolor='k', edgecolor='k'), 
                             #axes, 
                             title=title, 
                             #annotate, 
                             black_bg=True, 
                             #alpha, 
                             edge_kwargs={
                                          'alpha':0.8,
                                          'linewidth':9,
                                          }, 
                             node_kwargs={
                                          'edgecolors':'k',
                                          }, 
                             #colorbar=True
                             )


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


def print_connections(matrix, labels, fname):

    file_ = open(fname, 'w')
    
    upp_ind = np.triu_indices(matrix.shape[0], k=1)
    upp_matrix = matrix[upp_ind]
    sort_index = np.argsort(np.abs(upp_matrix))
    sort_matrix = upp_matrix[sort_index]
    ord_ind = tuple([i[sort_index] for i in upp_ind])
    
    for i, elem in enumerate(sort_matrix):
        file_.write(labels[ord_ind[0][i]]+','+labels[ord_ind[1][i]]+','+str(elem))
        file_.write('\n')
    
    file_.close()
    
    
    
def overlap_atlases():
    overlapping = []

    for n, net in enumerate(findlab):
        for sub_net in np.unique(net)[1:]:
            maskd = img.get_data() * (net == sub_net)
            over_roi = np.unique(maskd)[1:]
            print '%s %s: %s %s' % (str(n), str(sub_net), str(over_roi), collections.Counter(maskd[maskd!=0]))
            overlapping.append([n, sub_net, np.unique(maskd)[1:]])

#############################################

