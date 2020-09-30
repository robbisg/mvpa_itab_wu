# pylint: disable=maybe-no-member, method-hidden
import os
import nibabel as ni
import numpy as np
from mvpa_itab.conn.operations import copy_matrix
import itertools
from scipy.stats.stats import zscore



def network_connections(matrix, label, roi_list, method='within'):
    """
    Function used to extract within- or between-networks values
    """
    
    mask1 = roi_list == label
    
    if method == 'within':
        mask2 = roi_list == label
    else:
        mask2 = roi_list != label
    
    matrix_hori = np.meshgrid(mask1, mask1)[0] * np.meshgrid(mask2, mask2)[1]
    matrix_vert = np.meshgrid(mask1, mask1)[1] * np.meshgrid(mask2, mask2)[0]

    connections_ = matrix * (matrix_hori + matrix_vert)
    
    return connections_, matrix_hori + matrix_vert
    


def get_signed_connectome(matrix, method='negative'):
    """
    Function used to extract positive or negative values from matrix
    
    Parameters
    ----------
    matrix : 2D array, n x n
        Connectivity matrix in square form
    method : {'negative', 'positive'} | 'negative' default
        This is used to filter values in the connectomes.
    """
    
    sign = 1
    if method == 'negative':
        sign = -1
    
    mask_ = (matrix * sign) > 0
    signed_matrix = matrix * mask_
    
    return signed_matrix       


def aggregate_networks(matrix, roi_list, aggregation_fx=np.sum):
    """
    Function used to aggregate matrix values using 
    aggregative information provided by roi_list
    
    Parameters
    ----------
    matrix : numpy 2D array, shape n x n
        Connectivity matrix in squared form
    roi_list : list of string, length = n
        List of each ROI's network name. Each element represents
        the network that includes the ROI in that particular position.
        
    Returns
    -------
    aggregate_matrix : numpy 2D array, p x p
        The matrix obtained, by pairwise network sum 
        of nodes within networks.
        
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
        
        value = aggregation_fx(matrix * mask_roi)
        #value /= np.sum(mask_roi)
        
        aggregate_matrix[x, y] = value
    
    # Copy matrix in the lower part
    aggregate_matrix = copy_matrix(aggregate_matrix)
    
    # This is to fill the diagonal with within-network sum of elements
    for i, n in enumerate(unique_rois):
        
        diag_matrix, _ = network_connections(matrix, n, roi_list)
        aggregate_matrix[i, i] = aggregation_fx(diag_matrix) 
        # aggregate_matrix[i, i] = np.mean(diag_matrix) 
    
    return aggregate_matrix
        
        
def within_between(matrix, networks):
    """
    This function is used to extract from a connectivity matrix the mean value
    of within- and between-network correlation, for each network.
    
    Parameters
    ----------
    matrix : n x n numpy array
        The connectivity matrix used for the analysis.
    
    networks : n-lenght string array
        This array indicates which network the node is part of.
        
        
    Returns
    -------
    results : dict
        Returns a dictionary. Each item is composed by a key 
        representing the network name and a value which is a two elements list,
        first element is the between-network value, 
        the second is the within-network.
    """
        
        
        
    wb_results = dict()
    for network in np.unique(networks):
        wb_results[network] = list()
        for m_ in ['between', 'within']:
            net_, _ = network_connections(matrix, network, networks, method=m_)
            value_ = np.nanmean(net_[np.nonzero(net_)])
            wb_results[network].append(np.nan_to_num(value_))
    
    return wb_results        
        

def get_feature_weights_matrix(weights, sets, mask, indices=None):
    """
    Function used to compute the average weight matrix in case of
    several cross-validation folds and feature selection for each
    fold.
    
    Parameters
    ----------
    weights : ndarray shape=(n_folds,  n_selected_features)
        The weights matrix with the shape specified in the signature
    sets : ndarray shape=(n_folds, n_selected_features)
        This represents the index in the square matrix of the feature selected 
        by the algorithm in each cross-validation fold
    mask : ndarray shape=(n_roi, n_roi)
        The mask matrix of the valid ROIs selected. Important: this matrix
        should be triangular with the lower part set to zero.
    indices : tuple
        This is equal to np.nonzero(mask)
        
    Returns
    -------
    matrix: ndarray n_roi x n_roi
        It returns the average weights across cross-validation fold in
        square form.
    
    """
    
    if indices is None:
        indices = np.nonzero(mask)
    
    
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
