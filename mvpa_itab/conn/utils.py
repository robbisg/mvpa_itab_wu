# pylint: disable=maybe-no-member, method-hidden
import os
import nibabel as ni
import numpy as np
from mvpa_itab.conn.operations import copy_matrix
import itertools
from scipy.stats.stats import zscore



def find_roi_center(img, roi_value):
    """
    This function gives the x,y,z coordinates of a particular ROI using
    the given segmented image and the image level value used to select the ROI.
    
    Parameters
    ----------
    img : nibabel Nifti1Image instance
        The Nifti1Image instance of the segmented image.
    roi_value : int
        The value of the ROI as represented in the segmented image
        
    Returns
    -------
    xyz : tuple
        A triplets representing the xyz coordinate of the selected ROI.
    """
    
    
    affine = img.get_affine()
    
    mask_ = np.int_(img.get_data()) == roi_value
    ijk_coords = np.array(np.nonzero(mask_)).mean(1)
    
    xyz_coords = ijk_coords * affine.diagonal()[:-1] + affine[:-1,-1]
    
    return xyz_coords



def get_atlas90_coords():
    """
    Function used to obtain coordinates of the ROIs contained in the AAL90 atlas.
    The atlas used is the 2mm nifti version of the atlas.
    
    Returns
    -------
    coords : n x 3 numpy array
        The array containing n xyz coordinates in MNI space, one for each unique value of the atlas
    """
    atlas90 = ni.load('/media/robbis/DATA/fmri/templates_AAL/atlas90_mni_2mm.nii.gz')
    coords = [find_roi_center(atlas90, roi_value=i) for i in np.unique(atlas90.get_data())[1:]]
    
    return np.array(coords)



def get_findlab_coords():
    """
    Function used to obtain coordinates of the networks contained in the findlab atlas.
    The atlas used is the 2mm nifti version of the atlas.
    
    Returns
    -------
    coords : n x 3 numpy array
        The array containing n xyz coordinates in MNI space, one for each unique value of the atlas
    """
    roi_list = os.listdir('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/')
    roi_list.sort()
    findlab = [ni.load('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/'+roi) for roi in roi_list]
    f_coords = []
    for img_ in findlab:
        f_coords.append(np.array([find_roi_center(img_, roi_value=np.int(i)) for i in np.unique(img_.get_data())[1:]]))
        
    return np.vstack(f_coords)
           




def get_atlas_info(atlas_name, background='black'):
    
    """
    Utility function used to load informations about the atlas used
    
    Parameters
    ----------
    
    atlas_name : string | {'atlas90', 'findlab'}
        A string used to understand the atlas information used for plots.
        
    Returns
    -------
    names : list of string
        The list of ROI names.
    
    colors : list of string
        The list of colors used in other functions
    
    index_ : list of int
        How node values should be ordered if the atlas has another order
        (used to separate left/right in the atlas90)
        
    coords : list of tuple (x,y,z)
        Coordinates of the ROI center (used in plot_connectomics)
        
    networks : list of string
        The list of network names.

    """
    
    if atlas_name.find('atlas90') != -1 or atlas_name.find('20150') != -1:
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
        networks = names

    
    elif atlas_name.find('findlab') != -1 or atlas_name.find('2014') != -1:
        coords = get_findlab_coords()
        roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)
        networks = roi_list.T[-2]
        names = roi_list.T[2]
        
        dict_ = {'Auditory':'lightgray', 
                 'Basal_Ganglia':'honeydew', 
                 'LECN':'tomato',
                 'Language':'darkgrey', 
                 'Precuneus':'teal',
                 'RECN':'lightsalmon', 
                 'Sensorimotor':'plum', 
                 'Visuospatial':'slateblue', 
                 'anterior_Salience':'yellowgreen',
                 'dorsal_DMN':'lightsteelblue', 
                 'high_Visual':'khaki', 
                 'post_Salience':'mediumseagreen', 
                 'prim_Visual':'gold',
                 'ventral_DMN':'lightblue'
                 }
        """
        dict_ = {'Auditory':'silver', 
                 'Basal_Ganglia':'white', 
                 'LECN':'red',
                 'Language':'darkorange', 
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
        """
        if background == 'white':
            dict_['anterior_Salience'] = 'gray'
            dict_['Basal_Ganglia'] = 'black'
        
        
        colors_lr = np.array([dict_[r.T[-2]] for r in roi_list])
        index_ = np.arange(90)
        
    return names, colors_lr, index_, coords, networks

   
  

        
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
    
    return connections_, matrix_mask
    


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


def aggregate_networks(matrix, roi_list):
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
        
        value = np.sum(matrix * mask_roi)
        #value /= np.sum(mask_roi)
        
        aggregate_matrix[x, y] = value
    
    # Copy matrix in the lower part
    aggregate_matrix = copy_matrix(aggregate_matrix)
    
    # This is to fill the diagonal with within-network sum of elements
    for i, n in enumerate(unique_rois):
        
        diag_matrix, _ = network_connections(matrix, n, roi_list)
        aggregate_matrix[i, i] = np.sum(diag_matrix) 
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
        

def get_feature_weights_matrix(weights, sets, mask, indices):
    """
    Function used to compute the average weight matrix in case of
    several cross-validation folds and feature selection for each
    fold.
    
    Parameters
    ----------
    weights : ndarray shape n_folds x n_selected_features
        The weights matrix with the shape specified in the signature
    sets : ndarray shape n_folds x n_selected_features
        This represents the index in the square matrix of the feature selected 
        by the algorithm in each cross-validation fold
    mask : ndarray shape n_roi x n_roi 
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
