import statsmodels.api as sm
import itertools
import nibabel as ni
import numpy as np
import os
from sklearn.preprocessing.label import LabelEncoder
from theano.tensor.nnet.tests.test_conv3d2d import ndimage


def design_matrix(sample_labels):
    """
    Parameters
    ---------
    sample_labels: 
        a numpy matrix, for each sample a vector with condition
        which we would like to model.
        
    
    Returns
    -------
    X: the design matrix.
    factor_labels: the labels of the design-matrix columns
    factor_num : number of factors for each condition
    
    """
        
    factor_num = []
    n_factors = 0
    
    for i in range(sample_labels.shape[1]):
        unique_labels = np.unique(sample_labels[:,i])
        if len(unique_labels) == 1:
            label_factors = 0
        else:
            label_factors = len(unique_labels)
        
        n_factors+=label_factors
        factor_num.append(label_factors)
    
    X = np.zeros((sample_labels.shape[0], n_factors))
    
    lb = LabelEncoder()
    factor_labels = []
    offset = 0
    for i, factor in enumerate(factor_num):
        if factor == 0:
            continue
        index = lb.fit_transform(sample_labels.T[i])
        for j in range(sample_labels.shape[0]):
            X[j,index[j]+offset] = 1
        
        factor_labels.append(lb.classes_)
        
        offset+=factor
    
    return X, np.hstack(factor_labels), factor_num



def build_contrast(factor_num, factor_to_test, comparison_type="zero", const_value=0.):
    """
    comparison_type = "zero" --> f1=0, f2=0, f3=0 ecc.
                      "all" ---> f1=f2=f3=...=fn
                      
    This is only for one factor!
                      
    """
    
    if comparison_type == "zero":
        rows = factor_num[factor_to_test]
    elif comparison_type == "all":
        rows = factor_num[factor_to_test] - 1
        
    contrast = []
    for i in range(rows):
        c = np.zeros((np.sum(factor_num)))
        sum_ = 0
        for j, n_factor in enumerate(factor_num):
            
            if n_factor == 0:
                continue
            
            if j != factor_to_test:
                c[sum_:sum_+n_factor] = 1./n_factor
            else:
                c[sum_:sum_+n_factor] = get_value(i, n_factor, comparison_type)
                
            sum_+=n_factor
    
        contrast.append(c)
        
    const_array = np.zeros(rows)
    const_array[:] = const_value
        
    return np.vstack(contrast), const_array



def anova_ftest(data, design_matrix, contrast, const_term, mask_index):
    
    result_shape = list(data.shape[:-1]) + [2]
    result_map = np.zeros(result_shape)
    
    for x, y, z in mask_index:
        res = sm.OLS(data[x, y, z], design_matrix).transform()
        f_test = res.f_test((contrast, const_term))
        values_ = [f_test.fvalue.squeeze(), 1-f_test.pvalue]
        
        result_map[x,y,z] = np.array(values_)
        
    return result_map



def get_value(row, n_factor, comparison_type):
    
    c = np.zeros(n_factor)
    if comparison_type == "zero":
        c[row] = 1
    elif comparison_type == "all":
        c[row:row+2] = np.array([1, -1])
        
    return c




def get_rois(p_image, p_threshold, cluster_voxels_threshold, fill_holes=False):
    
    p_corrected = p_image.copy()
    p_corrected[p_corrected <= p_threshold] = 0
    
    label_im, n_labels = ndimage.label(p_corrected)
    
    output_img = np.zeros_like(label_im)
    j = 1
    for i in range(n_labels)[1:]:
        roi_mask = label_im == i
        
        if np.count_nonzero(roi_mask) <= cluster_voxels_threshold:
            output_img[roi_mask] = 0
        else:
            filled = roi_mask * j
            if fill_holes:
                filled = ndimage.binary_fill_holes(roi_mask).astype(int)
                filled *= j
                
            j += 1
            output_img += filled
    
    return output_img
                
            
            
        



