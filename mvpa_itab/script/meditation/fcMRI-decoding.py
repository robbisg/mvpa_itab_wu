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
from mvpa2.mappers.zscore import zscore
from mvpa_itab.conn.plot import get_plot_stuff, get_atlas_info,\
    plot_connectivity_circle_edited, plot_connectome
#from scipy.stats.stats import zscore

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
#results_dir = ['20140513_163451_connectivity_fmri']

for r in results_dir:
    print '··········· '+r+' ·············'
    conn = ConnectivityTest(path, subjects, r, roi_list)
    nan_mask = conn.get_results(['Samatha', 'Vipassana'])
    
    ds = conn.get_dataset()
    '''
    fx = mean_group_sample(['subjects', 'meditation'])
    ds = ds.get_mapped(fx)  
    '''
    clf = LinearCSVMC(C=10)
    # clf = RbfCSVMC()
    ds.targets = ds.sa.groups
    
    # ds.samples = decomposition.KernelPCA(kernel="poly", n_components=30).fit_transform(ds.samples)
    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                            FractionTailSelector(0.01,
                                                                mode='select',
                                                                tail='upper'))
    
    fclf = FeatureSelectionClassifier(clf, fsel)
    
    ds.samples = sscore(ds.samples, axis=1)
    # ds.samples = sscore(ds.samples, axis=0)
    # zscore(ds)
    cv = CrossValidation(fclf, 
                        NFoldPartitioner(cvtype=1), 
                        enable_ca=['stats'])
    results = dict()
    
    line_ += "%s ---\n" % (r)
    
    
    feature_selected = dict()
    feature_weights = dict()
    
    for med in ['Samatha','Vipassana']:
        #print '----------%s-------------' % (med)
        results[med] = []
        feature_selected[med] = np.zeros_like(ds.samples[0])
        feature_weights[med] = np.zeros_like(ds.samples[0])
        
        ds_med = ds[ds.sa.meditation == med]
        
        balancer = Balancer(count=100, apply_selection=True, limit=None)
        
        gen = balancer.generate(ds_med)
        
        balancer_res = []
        for i, ds_ in enumerate(gen):
            # print '---------'
            # ds_.samples = sscore(ds_.samples, axis=1)
            err = cv(ds_)
            #print cv.ca.stats
            try:
                sensana = fclf.get_sensitivity_analyzer()
            except NotImplementedError:
                results[med].append(1 - np.mean(err))
                line_ += "%s ---> %.2f\n" % (med, np.mean(results[med]))
                continue
            
            ds_sens = sensana(ds_)
            # Add numerosity to selected features
            feature_selected[med][np.nonzero(ds_sens.samples)[1]] += 1
            # Add weights to selected features
            feature_weights[med] += ds_sens.samples[0]
            
            #results[med].append(1 - np.mean(err))
            balancer_res.append(1 - np.mean(err))
            #print str(1 - np.mean(err))
        
        feature_weights[med] /= feature_selected[med]
        feature_weights[med][np.isnan(feature_weights[med])] = 0
        results[med].append(np.mean(balancer_res))
        line_ += "%s ---> %.4f\n" % (med, np.mean(balancer_res))
    

    
file_.write(line_)
file_.close()


for med, l_ in zip(['Samatha','Vipassana'], ['(FA)', '(OM)']):
    
    f_array = feature_selected[med].copy()
    w_array = feature_weights[med].copy()
    
    f_nz = f_array[np.nonzero(f_array)]
    
    # We selected only feature selected often
    threshold = f_nz.mean() + 0.5*f_nz.std()
    #f_array[f_array < threshold] = 0
    #w_array[f_array < threshold] = 0 # Weights selected based on chosen features
    
    # zscoring weights
    w_nz = w_array[np.nonzero(w_array)]
    w_nz = (w_nz - np.mean(w_nz))/np.std(w_nz)
    
    w_array[np.nonzero(w_array)] = w_nz
    
    f_matrix = copy_matrix(array_to_matrix(f_array, nan_mask), diagonal_filler=0)
    w_matrix = copy_matrix(array_to_matrix(w_array, nan_mask), diagonal_filler=0)
    
   
    names, colors, index_, coords = get_atlas_info('findlab')

    
    title_ = "%s %s" % (med, l_)
    # f_matrix[f_matrix == 0] = np.nan
    f, _ = plot_connectivity_circle_edited(f_matrix[index_][:,index_], 
                                            names[index_], 
                                            node_colors=colors[index_],
                                            node_size=f_matrix.sum(0)[index_]*3.5,
                                            con_thresh = 15.,
                                            title=title_,
                                            node_angles=circular_layout(names, 
                                                                        list(names),
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
    
    f.savefig(os.path.join(path, med+'_features.png'), facecolor='black')
    plot_connectome(f_matrix, 
                    coords, 
                    colors, 
                    3.5*f_matrix.sum(0)[index_],                    
                    15.,
                    os.path.join(path, med+'_features_brain.png'),
                    order=index_,
                    title=None,
                    max_=100.,
                    min_=20.,
                    #display_='ortho'
                    )
    
    vmax = np.max(np.abs(w_matrix))
    
    # w_matrix[w_matrix == 0] = np.nan
    
    f, _ = plot_connectivity_circle_edited(w_matrix[index_][:,index_], 
                                            names[index_], 
                                            node_colors=colors[index_],
                                            node_size=1.5*np.abs(w_matrix).sum(0)[index_]**2.5,
                                            con_thresh = 1.4,
                                            title=title_,
                                            node_angles=circular_layout(names, 
                                                                        list(names),
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
    
    f.savefig(os.path.join(path, med+'_weights.png'), facecolor='black')  
    plot_connectome(w_matrix, 
                    coords, 
                    colors, 
                    4.*np.abs(w_matrix).sum(0)[index_]**2.,
                    1.4,
                    os.path.join(path, med+'_weights_brain.png'),
                    order=index_,
                    title=None,
                    max_=3.,
                    min_=-3.,
                    cmap=pl.cm.bwr,
                    #display_='ortho'
                    )

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
        
               
        
        