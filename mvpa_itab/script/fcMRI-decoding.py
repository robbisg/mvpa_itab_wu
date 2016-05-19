from scipy.io import savemat
import numpy as np
import os
from mvpa_itab.connectivity import load_matrices, z_fisher, plot_matrix
from nitime.analysis import SeedCorrelationAnalyzer
from nitime.timeseries import TimeSeries
from scipy.io import loadmat
from scipy.stats import ttest_ind
from scipy.stats import zscore as sscore
from mvpa_itab.io.base import load_dataset
from mvpa_itab.conn.io import load_fcmri_dataset, copy_matrix
from mvpa2.clfs.svm import LinearCSVMC
from scipy.stats.stats import zscore

path = '/media/robbis/DATA/fmri/monks/0_results/'

results_dir = os.listdir(path)
 
results_dir = [r for r in results_dir if r.find('connectivity') != -1]
roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)


for r in results_dir:
    
    fname = os.path.join(path, r, 'zcorrelation_matrix.mat')
    fdata = loadmat(fname)
    
    data = fdata['z_matrix']
    
    nrun = data.shape[2]
    print nrun
    ds = load_fcmri_dataset(data, 
                            subjects.T[0], 
                            ['Samatha', 'Vipassana'], 
                            fdata['groups'], 
                            fdata['level'].squeeze(),
                            n_run=nrun)
    
    cv = CrossValidation(LinearCSVMC(C=1), 
                         NFoldPartitioner(cvtype=2), 
                         enable_ca=['stats'])
    
    
    ds.samples = sscore(ds.samples, axis=1)
    
    #zscore(ds, chunks_attr=None)
    
    ds.targets = ds.sa.group
    ds = ds[~np.logical_or(ds.sa.level == 300, ds.sa.level == 600)]
        
    #for g in np.unique(fdata['groups']):
    for g in ['Samatha','Vipassana']:
        print '---------------------------'
        print r+' ----- '+g
        err = cv(ds[ds.sa.meditation == g])
        print cv.ca.stats
        
##########################################
file_ = open(os.path.join('/media/robbis/DATA/fmri/monks/', '0_results', 'results_decoding_new.txt'), 'w')
line_ = ""
for r in results_dir:
    
    conn = ConnectivityTest(path, subjects, r, roi_list)
    conn.get_results(['Samatha', 'Vipassana'])
    
    ds = conn.get_dataset()
    '''
    fx = mean_group_sample(['subjects', 'meditation'])
    ds = ds.get_mapped(fx)  
    '''
    clf = LinearCSVMC(C=10)
    #clf = RbfCSVMC()
    ds.targets = ds.sa.groups
    
    #ds.samples = decomposition.KernelPCA(kernel="poly", n_components=30).fit_transform(ds.samples)
    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                            FractionTailSelector(0.01,
                                                                mode='select',
                                                                tail='upper'))
    
    fclf = FeatureSelectionClassifier(clf, fsel)
    
    ds.samples = sscore(ds.samples, axis=0)
    cv = CrossValidation(fclf, 
                        NFoldPartitioner(cvtype=1), 
                        enable_ca=['stats'])
    results = dict()
    
    line_ += "%s ---\n" % (r)
    
    
    feature_selected = dict()
    feature_weights = dict()
    
    for med in ['Samatha','Vipassana']:
        print '----------%s-------------' % (med)
        results[med] = []
        feature_selected[med] = np.zeros_like(ds.samples[0])
        feature_weights[med] = np.zeros_like(ds.samples[0])
        
        ds_med = ds[ds.sa.meditation == med]
        #zscore(ds_med, chunks_attr='subjects')  
        balancer = Balancer(count=100, apply_selection=True)
        
        gen = balancer.generate(ds_med)
        
        for i, ds_ in enumerate(gen):
            print '---------'
            
            err = cv(ds_)
            print cv.ca.stats
            try:
                sensana = fclf.get_sensitivity_analyzer()
            except NotImplementedError:
                results[med].append(1 - np.mean(err))
                line_ += "%s ---> %.2f\n" % (med, np.mean(results[med]))
                continue
            ds_sens = sensana(ds_)
            
            feature_selected[med][np.nonzero(ds_sens.samples)[1]] += 1
            feature_weights[med] += ds_sens.samples[0]
            
            results[med].append(1 - np.mean(err))
            
            #print str(1 - np.mean(err))
        
        feature_weights[med] /= feature_selected[med]
        feature_weights[med][np.isnan(feature_weights[med])] = 0

        line_ += "%s ---> %.2f\n" % (med, np.mean(results[med]))
    
file_.write(line_)
file_.close()


for med, l_ in zip(['Samatha','Vipassana'], ['(FA)', '(OM)']):
    
    f_array = feature_selected[med].copy()
    w_array = feature_weights[med].copy()
    
    f_nz = f_array[np.nonzero(f_array)]
    
    # We selected only feature selected often
    threshold = f_nz.mean() + 2*f_nz.std()
    f_array[f_array < threshold] = 0
    w_array[f_array < threshold] = 0 # Weights selected based on features
    
    # zscoring weights
    w_nz = w_array[np.nonzero(w_array)]
    w_nz = (w_nz - np.mean(w_nz))/np.std(w_nz)
    
    w_array[np.nonzero(w_array)] = w_nz
    
    f_matrix = copy_matrix(array_to_matrix(f_array), diagonal_filler=0)
    w_matrix = copy_matrix(array_to_matrix(w_array), diagonal_filler=0)
    
    # reorder labels
    names = roi_list[conn.network_names].T[1]
    names_inv = np.array([n[::-1] for n in names])
    index_ = np.argsort(names_inv)
    names_lr = names[index_]
    dict_ = {'L':'#89CC74', 'R':'#7A84CC'}
    colors_lr = np.array([dict_[n[:1]] for n in names_inv])[index_]
    
    
    names_lr = np.array([n.replace('_', ' ') for n in names_lr])
    
    #names = conn.network_names
    # plot graphs
    
    title_ = "%s %s" % (med, l_)
    f_matrix[f_matrix == 0] = np.nan
    f, _ = plot_connectivity_circle(f_matrix[index_], 
                                    names_lr, 
                                    node_colors=colors_lr, 
                                    title=title_,
                                    node_angles=circular_layout(names_lr, list(names_lr)),
                                    fontsize_title=19,
                                    fontsize_names=13,
                                    fontsize_colorbar=13,
                                    colorbar_size=0.3,
                                    colormap='summer',
                                    #vmin=40,
                                    fig=pl.figure(figsize=(13,13))
                                    )
    
    f.savefig(os.path.join(path, med+'_features.png'), facecolor='black')
    
    vmax = np.max(np.abs(w_matrix))
    
    w_matrix[w_matrix == 0] = np.nan
    
    f, _ = plot_connectivity_circle(w_matrix[index_], 
                                    names_lr, 
                                    node_colors=colors_lr, 
                                    title=title_,
                                    node_angles=circular_layout(names_lr, list(names_lr)),
                                    fontsize_title=19,
                                    fontsize_names=13,
                                    fontsize_colorbar=13,
                                    colorbar_size=0.3,
                                    vmax=vmax,
                                    vmin=-vmax,
                                    colormap='bwr',
                                    fig=pl.figure(figsize=(13,13))
                                    )
    
    f.savefig(os.path.join(path, med+'_weights.png'), facecolor='black')  


#### Permutation ###

r = ''

conn = ConnectivityTest(path, subjects, r, roi_list)
conn.get_results(['Samatha', 'Vipassana'])

ds = conn.get_dataset()
'''
fx = mean_group_sample(['subjects', 'meditation'])
ds = ds.get_mapped(fx)  
'''
#clf = LinearCSVMC(C=10)

skclf = SVC(C=10, kernel='linear', class_weight='balanced')
clf = SKLLearnerAdapter(skclf)   


ds.targets = ds.sa.groups

fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                        FractionTailSelector(0.01,
                                                            mode='select',
                                                            tail='upper'))

fclf = FeatureSelectionClassifier(clf, fsel)
ds.samples = sscore(ds.samples, axis=1)
cv = CrossValidation(fclf, 
                    NFoldPartitioner(cvtype=2), 
                    enable_ca=['stats'])
results_weight = dict()


n_permutations = 2000
 
for med in ['Samatha','Vipassana']:
    #print '----------%s-------------' % (med)
    results_weight[med] = []
    
    ds_med = ds[ds.sa.meditation == med]
    #zscore(ds_med, chunks_attr='subjects')  
    '''
    balancer = Balancer(count=1, apply_selection=True)#, limit='subjects')
    gen = balancer.generate(ds_med)
    ds_ = gen.next()
    for i in range(n_permutations):
        #print '---------'
        ds_.targets = permutation(ds_.targets)
        err = cv(ds_)
        results_weight[med].append(1 - np.mean(err))
    '''    
        #print cv.ca.stats
    err = cv(ds_med)
    print cv.ca.stats
        
               
        
        