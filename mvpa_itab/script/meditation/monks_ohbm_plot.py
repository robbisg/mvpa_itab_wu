from mvpa_itab.conn.io import ConnectivityLoader
from mvpa_itab.conn.utils import copy_matrix, get_plot_stuff, aggregate_networks
from mvpa_itab.conn.plot import *
import nibabel as ni
from scipy.stats.stats import zscore
import matplotlib.pyplot as pl
import os



    

def get_feature_selection_matrix(feature_set, n_features, mask):
    
    h_values_, _ = np.histogram(feature_set.flatten(), 
                                bins=np.arange(0, n_features+1))
    
    
    mask = np.triu(mask, k=1)
    mask_indices = np.nonzero(mask)
    mask[mask_indices] = h_values_
    
    return np.nan_to_num(copy_matrix(mask, diagonal_filler=0))


def get_feature_weights_matrix(weights, sets, mask, indices):
    
    weights = weights.squeeze()
    filling_vector = np.zeros(np.count_nonzero(mask))
    counting_vector = np.zeros(np.count_nonzero(mask))
    
    for s, w in zip(sets, weights):
        filling_vector[s] += zscore(w)
        counting_vector[s] += 1
        
    avg_weigths = np.nan_to_num(filling_vector/counting_vector)
    mask[indices] = avg_weigths    
    matrix = np.nan_to_num(copy_matrix(mask, diagonal_filler=0))
    
    return matrix



def get_node_size(matrix, absolute=True):
    
    matrix = np.nan_to_num(copy_matrix(matrix, diagonal_filler=0))
    if absolute == True:
        matrix = np.abs(matrix)
    
    return matrix.sum(axis=0)




  
    
    
def load_results(path, directory, condition, result_type='values'):
    
    if result_type == 'both':
        values = ['values', 'permutation']
        
    else:
        values = [result_type]
    
    output = dict()
    for t_ in values:
        fname_ = os.path.join(path, directory, condition+'_%s_1000_cv_50.npz' % (t_))
        results_ = np.load(fname_)
        output[t_] = results_
    
    return output


def get_analysis_mask(path, subjects, directory, roi_list):
    ######## Get matrix infos ###############
    conn_test = ConnectivityLoader(path, 
                                   subjects, 
                                   directory, 
                                   roi_list)
    
    # Get nan mask to correctly fill matrix
    nan_mask = conn_test.get_results(['Samatha', 'Vipassana'])
    # Transform matrix into float of ones
    mask_ = np.float_(~np.bool_(nan_mask))
    # Get the upper part of the matrix
    mask_ = np.triu(mask_, k=1)
    
    return mask_


    
def write_results(directory_list, conditions, n_permutations=1000.):
    
    res_path = '/media/robbis/DATA/fmri/monks/0_results/'
    path = '/media/robbis/DATA/fmri/monks/'
    roi_list = []
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                          delimiter=',',
                          dtype=np.str)
    
    subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)
    
    for dir_ in directory_list:
        for cond_ in conditions:
            
            results_ = load_results(path, dir_, cond_, 'both')
            
            values_ = results_['values']['arr_0'].tolist()
            errors_ = values_['errors_']
            sets_ = values_['sets_']
            weights_ = values_['weights_']
            samples_ = values_['samples_']
            

            values_p = results_['permutation']['arr_0'].tolist()
            errors_p = values_p['errors_p']
            sets_p = values_p['sets_p']
            weights_p = values_p['weights_p']
            samples_p = values_p['samples_p']
            
            errors_p = np.nanmean(errors_p, axis=1)
            
            n_permutations = np.float(errors_p[:,0].shape[0])
                        
            print('-----------'+dir_+'-------------')
            print(cond_)
            print('MSE = '+str(errors_[:,0].mean())+' -- p '+ \
                str(np.count_nonzero(errors_p[:,0] < errors_[:,0].mean())/n_permutations))
            print('COR = '+str(np.nanmean(errors_[:,1]))+' -- p '+ \
                str(np.count_nonzero(errors_p[:,1] > np.nanmean(errors_[:,1]))/n_permutations))
                
            directory_ = dir_
            learner_ = "SVR_C_1" 
        
            prename = "%s_%s" %(cond_, learner_)
            
            ######## Get matrix infos ###############
            
            mask_ = get_analysis_mask(path, subjects, dir_, roi_list)
            
            mask_indices = np.nonzero(mask_)
            n_features = np.count_nonzero(mask_)
            
            
            
            ###### Plot of distributions of errors and permutations #########
            
            plot_regression_errors(errors_, 
                                   errors_p, 
                                   os.path.join(res_path, dir_), 
                                   prename=prename, 
                                   errors_label=['MSE','COR'])
                        
            
            ##### Plot of connection distributions ########
            
            plot_features_distribution(sets_, 
                                       sets_p, 
                                       os.path.join(res_path, dir_), 
                                       prename=prename, 
                                       n_features=n_features)
            
            
            ######## Plot weights connectomics ###########
            
            matrix_ = get_feature_weights_matrix(weights_, sets_, mask_, mask_indices)
                    
            size_w = get_node_size(matrix_)
            
            names_lr, colors_lr, index_, coords, networks = get_atlas_info(dir_)
            
            plot_connectomics(matrix_, 
                              2*size_w**2, 
                              os.path.join(res_path, dir_), 
                              prename, 
                              colormap='bwr',
                              vmin=-3.,
                              vmax=3.,
                              name='weights_new',
                              title=cond_,
                              save=False,
                              node_colors=colors_lr,
                              node_names=names_lr,
                              node_order=index_,
                              node_coords=coords,
                              networks=networks
                              )
            
            
            
            
                                        
            ######### Plot choice connectomics #################
            
            matrix_ = get_feature_selection_matrix(sets_, 
                                                   n_features, 
                                                   mask=np.float_(~np.bool_(mask_)))
            
            size_f = get_node_size(matrix_)
            
            plot_connectomics(matrix_, 
                              size_f*4.5, 
                              os.path.join(res_path, dir_), 
                              prename, 
                              threshold=15,
                              vmin=0.,
                              vmax=50.,
                              name='choice_new',
                              title=cond_,
                              save=False,
                              node_colors=colors_lr,
                              node_names=names_lr,
                              node_order=index_,
                              node_coords=coords,
                              networks=networks
                              )
                                        
            pl.close('all')   
    
###################################################




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


def bottom_up_script(directory, condition):
    
    subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)
    
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                          delimiter=',',
                          dtype=np.str)
    
    
    path = '/media/robbis/DATA/fmri/monks/0_results/'
    results_ = load_results(path, directory, condition, 'values')
    values_ = results_['values']['arr_0'].tolist()
    mask_ = get_analysis_mask(path, subjects, directory, roi_list)
            
    mask_indices = np.nonzero(mask_)
    n_features = np.count_nonzero(mask_)
    
    w_matrix = get_feature_weights_matrix(values_['weights_'], 
                                          values_['sets_'], 
                                          mask_, 
                                          mask_indices)
    
    w_aggregate = aggregate_networks(w_matrix, roi_list.T[-2])
    
    names_lr, colors_lr, index_, coords, networks = get_atlas_info(directory)
    
    _, idx = np.unique(networks, return_index=True)
    
    plot_connectomics(w_aggregate, 
                      2*np.abs(w_aggregate.sum(axis=1))**2, 
                      os.path.join(path, directory), 
                      condition+'_aggregate_weights_regression', 
                      save=True,
                      colormap='bwr',
                      vmin=-10.,
                      vmax=10.,
                      node_names=np.unique(networks),
                      node_colors=colors_lr[idx],
                      node_coords=coords[idx],
                      node_order=np.arange(0, len(idx)),
                      networks=np.unique(networks)                      
                      )
    
    
    
    
    
    
    
    
    