import numpy as np
import os

from nitime.analysis import SeedCorrelationAnalyzer
from nitime.timeseries import TimeSeries

from scipy.io import loadmat, savemat
from scipy.stats import ttest_ind
from scipy.stats import zscore as sscore

from mvpa_itab.io.base import load_dataset
from mvpa2.suite import *

from mvpa_itab.conn.io import load_fcmri_dataset,ConnectivityLoader
from mvpa_itab.conn.operations import copy_matrix, array_to_matrix
from mvpa_itab.conn.connectivity import load_matrices, z_fisher
from mvpa_itab.conn.plot import get_atlas_info, plot_connectome, \
                plot_connectivity_circle_edited, plot_connectomics
from mvpa_itab.conn.utils import aggregate_networks, get_signed_connectome,\
    network_connections
from sklearn.svm.classes import SVC

import cPickle as pickle

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
results_dir = ['20140513_163451_connectivity_fmri']
#results_dir = ['20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri']

for r in results_dir:
    print '··········· '+r+' ·············'
    conn = ConnectivityLoader(path, subjects, r, roi_list)
    nan_mask = conn.get_results(['Samatha', 'Vipassana'])
    #nan_mask = conn.get_results(['Rest'])
    ds = conn.get_dataset()
    '''
    fx = mean_group_sample(['subjects', 'meditation'])
    ds = ds.get_mapped(fx)  
    '''
    clf = LinearCSVMC(C=1)
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
    
    for med in ['Samatha', 'Vipassana']:
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

directory = r
save_path = path
wb_meditation = dict()

feature_selected = pickle.load(open(os.path.join(path, 
                                                 r, 
                                                 'feature_selected_500_ds_balancing_final.obj'),
                                    'r'))
feature_weights = pickle.load(open(os.path.join(path, 
                                                r, 
                                                'feature_weights_500_ds_balancing_final.obj'),
                                   'r'))

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
        
    title = "%s %s" % (med, l_)
    # f_matrix[f_matrix == 0] = np.nan
    ##################################################################################
    condition = med
    w_aggregate = aggregate_networks(w_matrix, roi_list.T[-2])
    
    names_lr, colors_lr, index_, coords, networks = get_atlas_info('findlab')
    
    _, idx = np.unique(networks, return_index=True)
    
    
    ##########################################################################
    
    plot_connectomics(w_aggregate, 
                      5*np.abs(w_aggregate.sum(axis=1))**2, 
                      save_path=os.path.join(path, directory), 
                      prename=condition+'_aggregate_weights_decoding', 
                      save=False,
                      colormap='bwr',
                      vmin=-1*w_aggregate.max(),
                      vmax=w_aggregate.max(),
                      node_names=np.unique(networks),
                      node_colors=colors_lr[idx],
                      node_coords=coords[idx],
                      node_order=np.arange(0, len(idx)),
                      networks=np.unique(networks),
                      title=title                   
                      )
    
    plot_connectomics(w_matrix,
                      5*np.abs(w_matrix.sum(axis=1))**2, 
                      save_path=os.path.join(path, directory), 
                      prename=condition+'_weights_decoding', 
                      save=False,
                      colormap='bwr',
                      vmin=w_matrix.max()*-1,
                      vmax=w_matrix.max(),
                      node_names=names_lr,
                      node_colors=colors_lr,
                      node_coords=coords,
                      node_order=index_,
                      networks=networks,
                      threshold=1.4,
                      title=title             
                      )
    
    for method in ['positive', 'negative']:
        aggregate = get_signed_connectome(w_aggregate, method=method)
        colormap_ = 'bwr'
    
        plot_connectomics(aggregate, 
                          5*np.abs(aggregate.sum(axis=1))**2, 
                          save_path=os.path.join(path, directory), 
                          prename=condition+'_'+method+'_aggregate_weights_decoding', 
                          save=True,
                          colormap=colormap_,
                          vmin=np.abs(aggregate).max()*-1,
                          vmax=np.abs(aggregate).max(),
                          node_names=np.unique(networks),
                          node_colors=colors_lr[idx],
                          node_coords=coords[idx],
                          node_order=np.arange(0, len(idx)),
                          networks=np.unique(networks),
                          threshold=1.,
                          title=title+' '+method+' aggregate'                  
                          )
    

    
    
        sign_matrix = get_signed_connectome(w_matrix, method=method)
        plot_connectomics(sign_matrix,
                          5*np.abs(sign_matrix.sum(axis=1))**2, 
                          save_path=os.path.join(path, directory), 
                          prename=condition+'_'+method+'_weights_decoding', 
                          save=True,
                          colormap=colormap_,
                          vmin=np.abs(sign_matrix).max()*-1,
                          vmax=np.abs(sign_matrix).max(),
                          node_names=names_lr,
                          node_colors=colors_lr,
                          node_coords=coords,
                          node_order=index_,
                          networks=networks,
                          threshold=1.2,
                          title=title+' '+method         
                          )
    

    pl.close('all')
    ##################################################################
    within_between = dict()
    for network in np.unique(networks):
        within_between[network] = list()
        for m_ in ['between', 'within']:
            net_, mask_ = network_connections(w_matrix, network, networks, method=m_)
            value_ = np.nanmean(net_[np.nonzero(net_)])
            within_between[network].append(np.nan_to_num(value_))
        

    wb_meditation[med] = within_between


###########################
from matplotlib import colors

_, idxnet = np.unique(networks, return_index=True)
_, idx = np.unique(colors_lr, return_index=True)
color_net = dict(zip(networks[np.sort(idxnet)], colors_lr[np.sort(idx)]))

for meditation, connections in wb_meditation.iteritems():
    
    pl.figure(figsize=(13.2,10), dpi=200)
    
    keys = np.sort(connections.keys())
    values = list()
    for k_, v_ in connections.iteritems():
        values.append(v_)
        lines_ = [pl.plot(v_, 'o-', c=color_net[k_], 
                          markersize=20, linewidth=5, alpha=0.6, 
                          label=k_)]
         
    values = np.array(values)
    
    pl.legend()
    pl.ylabel("Average connection weight")
    pl.xticks([0,1,1.4], ['Between-Network', 'Within-Network',''])
    pl.title(meditation+' within- and between-networks average weights')
    pl.savefig(os.path.join(path, directory,meditation+'_within_between.png'),
               dpi=200)
    




#### Permutation ###


r = ''

conn = ConnectivityLoader(path, subjects, r, roi_list)
conn.get_results(['Rest'])

ds = conn.get_dataset()
'''
fx = mean_group_sample(['subjects', 'meditation'])
ds = ds.get_mapped(fx)  
'''
clf = LinearCSVMC(C=1) # C=10

#skclf = SVC(C=10, kernel='linear', class_weight='balanced')
#clf = SKLLearnerAdapter(skclf)   


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
 
for med in ['Rest']:
    #print '----------%s-------------' % (med)
    results_weight[med] = []
    
    ds_med = ds[ds.sa.meditation == med]
    #zscore(ds_med, chunks_attr='subjects')  
    
    balancer = Balancer(count=1, apply_selection=True)#, limit='subjects')
    gen = balancer.generate(ds_med)
    ds_ = gen.next()
    for i in range(n_permutations):
        #print '---------'
        ds_.targets = np.random.permutation(ds_.targets)
        err = cv(ds_)
        results_weight[med].append(1 - np.mean(err))
       
        #print cv.ca.stats
    #err = cv(ds_med)
    #print cv.ca.stats
        
               
        
        