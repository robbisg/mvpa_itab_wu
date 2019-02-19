import statsmodels.api as sm
import itertools
import nibabel as ni
import numpy as np
import os
from sklearn.preprocessing.label import LabelEncoder
from sklearn.utils.extmath import cartesian


def design_matrix(sample_labels, interaction_indices=None):
    """
    Parameters
    ---------
    sample_labels: 
        a numpy matrix, for each sample a vector with the conditions
        which we would like to model.
        cols represent the type of conditions we want to model,
        row represent a combination of conditions that are represented by the row-variable.
        if we have a 2x3 design we build this matrix:
        [[0,0],
         [0,1],
         [0,2],
         [1,0],
         [1,1],
         [1,2]]
        
        
    
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
    
    n_interactions = 0
    if interaction_indices != None:
        interaction_factors = np.array(factor_num)[[interaction_indices]]
        n_interactions = np.prod(interaction_factors)
        Xint = np.zeros((sample_labels.shape[0], n_interactions))
    
    
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
        
        offset += factor
    
    if interaction_indices != None:
        interaction_product = [np.arange(v).tolist() for v in interaction_factors]
        interaction_gen = cartesian(interaction_product)
        
        # This is buggy!!
        Xint = np.zeros((sample_labels.shape[0], n_interactions))
        offset = interaction_indices[0] * np.sum(factor_num[:interaction_indices[0]])
        offset = np.int(offset)
        for i, int_indices in enumerate(interaction_gen):
            
            index1 = offset + int_indices[0]
            index2 = offset + int_indices[1] + factor_num[interaction_indices[0]]
            
            Xint[:,i] = X[:,index1] * X[:,index2]
            
            factor1 = interaction_indices[0]
            factor2 = interaction_indices[1]

            new_label = factor_labels[factor1][int_indices[0]] + "_" + \
                        factor_labels[factor2][int_indices[1]]
                        
            factor_labels.append(new_label)
        
        X = np.hstack((X, Xint))
        
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
    failed = 0
    for x, y, z in mask_index:
        
        res = sm.OLS(data[x, y, z], design_matrix).fit()
        try:
            f_test = res.f_test(contrast)
        except Exception as e:
            failed += 1
            continue
        values_ = [f_test.fvalue.squeeze(), 1 - f_test.pvalue]
        
        result_map[x,y,z] = np.array(values_)
    
    print(failed)
      
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
                
            
            
        



