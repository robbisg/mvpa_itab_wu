from mvpa_itab.conn.io import ConnectivityLoader
from mvpa_itab.conn.utils import copy_matrix, get_plot_stuff, aggregate_networks
from mvpa_itab.conn.plot import *
import nibabel as ni
from scipy.stats.stats import zscore
import matplotlib.pyplot as pl
import os


def plot_regression_errors(errors, permutation_error, save_path, prename='distribution', errors_label=['MSE','COR']):
    
    fig_ = pl.figure()
    bpp = pl.boxplot(permutation_error, showfliers=False, showmeans=True, patch_artist=True)
    bpv = pl.boxplot(errors, showfliers=False, showmeans=True, patch_artist=True)
    fname = "%s_perm_1000_boxplot.png" %(prename)
   
    
    for box_, boxp_ in zip(bpv['boxes'], bpp['boxes']):
        box_.set_facecolor('lightgreen')
        boxp_.set_facecolor('lightslategrey')
      
      
    pl.xticks(np.array([1,2]), errors_label)
    
    pl.savefig(os.path.join(save_path, fname))
    pl.close()
    
    return fig_


def plot_features_distribution(feature_set, 
                               feature_set_permutation, 
                               save_path, 
                               prename='features', 
                               n_features=90, 
                               n_bins=20):
    
    pl.figure()
    h_values_p, _ = np.histogram(feature_set_permutation.flatten(), 
                                 bins=np.arange(0, n_features+1))
    
    pl.hist(zscore(h_values_p), bins=n_bins)
    
    fname = "%s_features_set_permutation_distribution.png" % (prename)
    pl.savefig(os.path.join(save_path, 
                            fname))
    
    pl.figure()
    h_values_, _ = np.histogram(feature_set.flatten(), 
                                bins=np.arange(0, n_features+1))
    pl.plot(zscore(h_values_))
        
    
    fname = "%s_features_set_cross_validation.png" % (prename)
    pl.savefig(os.path.join(save_path, 
                            fname))
    
    pl.close('all')
    

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



def plot_connectomics(matrix, 
                      node_size, 
                      save_path, 
                      prename,
                      save=False,
                      **kwargs
                      ):
    
    
    
    _plot_cfg = {
                 'threshold':1.4,
                 'fontsize_title':19,
                 'fontsize_colorbar':13,
                 'fontsize_names':13,
                 'colorbar_size':0.3,
                 'colormap':'hot',
                 'vmin':-3,
                 'vmax':3,
                 'figure':pl.figure(figsize=(16,16)),
                 'facecolor':'black',
                 'dpi':150,
                 'name':'weights',
                 'title':'Connectome'               
                }
    
    
    
    _plot_cfg.update(kwargs)
     
    directory_ = save_path[save_path.rfind('/')+1:]
    
    #names_lr, colors_lr, index_, coords = get_plot_stuff(directory_)
    
    names_lr = kwargs['node_names']
    colors_lr = kwargs['node_colors']
    index_ = kwargs['node_order']
    coords = kwargs['node_coords']
    networks = kwargs['networks']
    
    matrix = matrix[index_][:,index_]
    names_lr = names_lr[index_]
    node_colors = colors_lr[index_]
    node_size = node_size[index_]
    
    
    f, _ = plot_connectivity_circle_edited(matrix, 
                                            names_lr, 
                                            node_colors=node_colors,
                                            node_size=node_size,
                                            con_thresh=_plot_cfg['threshold'],
                                            title=_plot_cfg['title'],
                                            node_angles=circular_layout(names_lr, 
                                                                        list(names_lr),
                                                                        ),
                                            fontsize_title=_plot_cfg['fontsize_title'],
                                            fontsize_names=_plot_cfg['fontsize_names'],
                                            fontsize_colorbar=_plot_cfg['fontsize_colorbar'],
                                            colorbar_size=_plot_cfg['colorbar_size'],
                                            colormap=_plot_cfg['colormap'],
                                            vmin=_plot_cfg['vmin'],
                                            vmax=_plot_cfg['vmax'],
                                            fig=_plot_cfg['figure'],
                                            )
            
    if save == True:
        fname = "%s_features_%s.png" % (prename, _plot_cfg['name'])
        
        f.savefig(os.path.join(save_path, fname),
                          facecolor=_plot_cfg['facecolor'],
                          dpi=_plot_cfg['dpi'])
    
    
    for d_ in ['x', 'y', 'z']:
        
        fname = None
        if save == True:
            fname = "%s_connectome_feature_%s_%s.png" %(prename, _plot_cfg['name'], d_)
            fname = os.path.join(save_path, fname)
            
        plot_connectome(matrix, 
                        coords, 
                        colors_lr, 
                        node_size,
                        _plot_cfg['threshold'],
                        fname,
                        cmap=_plot_cfg['colormap'],
                        title=None,
                        display_=d_,
                        max_=_plot_cfg['vmax'],
                        min_=_plot_cfg['vmin']
                        )
        
    
    f = plot_matrix(matrix, _, networks)
    if save == True:
        fname = "%s_matrix_%s.png" %(prename, _plot_cfg['name'])
        f.savefig(os.path.join(save_path, fname),
                          #facecolor=_plot_cfg['facecolor'],
                          dpi=_plot_cfg['dpi'])
  
    
    
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
            
            matrix_ = get_feature_selection_matrix(sets_, n_features, mask=np.float_(~np.bool_(nan_mask)))
            
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
    
    
    
    
    
    
    
    
    