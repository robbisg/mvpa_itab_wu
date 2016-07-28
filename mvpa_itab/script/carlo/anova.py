import itertools
import nibabel as ni

path = '/media/robbis/DATA/fmri/memory/0_results/balanced_analysis/local/'

levels = [1,3,5]
conditions = ['memory']#, 'memory']

groups = ['group1', 'group2']

product_iter = itertools.product(conditions, groups, levels)

total_ = []
design_matrix = []
n_factors = len(levels) + len(conditions) + len(groups)

for cond_, group_, lev_ in product_iter:
    
    fname = "%s_%s_evidence_%s_total.nii.gz" % (group_, cond_, str(lev_))
    print fname
    
    img_ = ni.load(os.path.join(path, fname))
    
    total_.append(img_.get_data())
    
    matrix_ = np.zeros((img_.shape[-1], n_factors))
    
    index_cond = np.nonzero(np.array(conditions) == cond_)[0][0]
    index_group = np.nonzero(np.array(groups) == group_)[0][0] + len(conditions)
    index_level = np.nonzero(np.array(levels) == lev_)[0][0] + len(conditions) + len(groups)
    
    matrix_[:, index_cond] = 1
    matrix_[:, index_group] = 1
    matrix_[:, index_level] = 1
    
    design_matrix.append(matrix_)

design_matrix = np.vstack(design_matrix)
total_matrix = np.concatenate(total_, axis=3)

mask_ = total_matrix.mean(3) > 0.05
mask_index = np.array(np.nonzero(mask_)).T
threshold_ = total_matrix[mask_].mean()
#threshold_ = 0.5

import statsmodels.api as sm

contrast = [[1,0.5,0.5,1,0,0], [1,0.5,0.5,0,1,0],[1,0.5,0.5,0,0,1]]
#const_terms = [0.5, 0.5, 0.5] # chance level
const_terms = [threshold_, threshold_, threshold_] 

result_shape = list(img_.shape[:-1]) + [2]
result_map = np.zeros(result_shape)

for x, y, z in mask_index:
    res = sm.OLS(total_matrix[x, y, z], design_matrix).fit()
    f_test = res.f_test((contrast, const_terms))
    values_ = [f_test.fvalue.squeeze(), f_test.pvalue]
    
    result_map[x,y,z] = np.array(values_)
    
fname = "_%s_omnibus_threshold_%s.nii.gz" % (cond_, str(threshold_)[:5])
ni.save(ni.Nifti1Image(result_map, img_.get_affine()),os.path.join(path, fname))
    
    


