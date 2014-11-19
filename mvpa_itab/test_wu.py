#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

from main_wu import *
from io import *
from mvpa2.clfs.transerror import ConfusionMatrix
import os
import copy
from similarity import *
    
    
def test_spatiotemporal(path, subjects, conf_file, type, **kwargs):
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
        if arg == 'balance':
            balance = kwargs[arg]
    
    total_results = dict()
    data_path = conf['data_path']
    for subj in subjects:
        try:
            ds = load_dataset(data_path, subj, type, **conf)
        except Exception, err:
            print err
            continue
        ds = preprocess_dataset(ds, type, **conf)
        
        if 'balance' in locals() and balance == True:
            if conf['label_included'] == 'all' and \
                conf['label_dropped'] == 'none':
                ds = balance_dataset(ds, 'fixation')
        
        r = spatiotemporal(ds, **conf)
        
        total_results[subj] = r
    
    conf['analysis_type'] = 'spatiotemporal'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)
    
    save_results(path, total_results, conf)
    
    return total_results

def test_spatial(path, subjects, conf_file, type, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    
    data_path = conf['data_path']
    
    for subj in subjects:
        print '------'
        try:
            ds = load_dataset(data_path, subj, type, **conf)
        except Exception, err:
            print err
            continue
        
        ds = preprocess_dataset(ds, type, **conf)
        
        r = spatial(ds, **conf)
        
        total_results[subj] = r
        
        #del ds
        
    
    conf['analysis_type'] = 'spatial'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    save_results(path, total_results, conf)
    
    return total_results


def test_clustering(path, subjects, analysis, conf_file, source='task', **kwargs):    
    
    if source == 'task':
        target = 'rest'
    else:
        target = 'task'
     
    conf_src = read_configuration(path, conf_file, source)
    conf_tar = read_configuration(path, conf_file, target)
    
    ##############################################
    conf_src['label_included'] = 'all'
    conf_src['label_dropped'] = 'none'
    conf_src['mean_samples'] = 'True'
    ##############################################
    for arg in kwargs:
        conf_src[arg] = kwargs[arg]
        conf_tar[arg] = kwargs[arg]
        
    total_results = dict()
    
    data_path = conf_src['data_path']
    
    for subj in subjects:
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
        
        if conf_src['label_included'] == 'all' and \
                conf_src['label_dropped'] != 'fixation':
                ds_src = balance_dataset(ds_src, 'fixation')
        
        r = clustering_analysis(ds_src, ds_tar, analysis, **kwargs)
        
        total_results[subj] = r
        
    conf_src['analysis_type'] = 'clustering'
    conf_src['analysis_task'] = 'task'
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)
    
    save_results(path, total_results, conf_src)

    return total_results


def test_transfer_learning(path, subjects, analysis,  conf_file, source='task', \
                           analysis_type='single', calculateSimilarity='True', **kwargs):
    
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    
    if source == 'saccade':
        target = 'face'
    else:
        if source == 'face':
            target = 'saccade'
    
    p = kwargs['p']
    ##############################################    
    ##############################################
    ##    conf_src['label_included'] = 'all'    ##   
    ##    conf_src['label_dropped'] = 'none'    ##
    ##    conf_src['mean_samples'] = 'False'    ##
    ##############################################
    ##############################################

    if analysis_type == 'group':
        
        if path.__class__ == conf_file.__class__ == list:  
            ds_src, _, conf_src = sources_merged_ds(path, subjects, conf_file, source, **kwargs)
            ds_tar, subjects, conf_tar = sources_merged_ds(path, subjects, conf_file, target, **kwargs)
            
            conf_src['permutations'] = 0
            conf_tar['permutations'] = 0
        else:
            print 'In group analysis path, subjects and conf_file must be lists: \
                    Check configuration file and/or parameters!!'
            return 0
    
    else:
        
        conf_src = read_configuration(path, conf_file, source)
        conf_tar = read_configuration(path, conf_file, target)
    
        for arg in kwargs:
            conf_src[arg] = kwargs[arg]
            conf_tar[arg] = kwargs[arg]
        
        
        data_path = conf_src['data_path']
    
    
    for arg in conf_src:
        if arg == 'map_list':
            map_list = conf_src[arg].split(',')
        if arg == 'p_dist':
            p = float(conf_src[arg])
            print p
    
    
    total_results = dict()
    
    for subj in subjects:
        print '-----------'
        
        if (len(subjects) > 1) or (subj != 'group'):
            try:
                ds_src = load_dataset(data_path, subj, source, **conf_src)
                ds_tar = load_dataset(data_path, subj, target, **conf_tar)
            except Exception, err:
                print err
                continue
         
            #Evaluate if is correct to do further normalization after merging two ds. 
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 

        if conf_src['label_included'] == 'all' and \
           conf_src['label_dropped'] != 'fixation':
                print 'Balancing dataset...'
                ds_src = balance_dataset(ds_src, 'fixation')        
        
        r = transfer_learning(ds_src, ds_tar, analysis, **conf_src)
        
        
        
        pred = np.array(r['classifier'].ca.predictions)
        targets = r['targets']
        
        c_m = ConfusionMatrix(predictions=pred, targets=targets)
        c_m.compute()
        r['confusion_target'] = c_m
        print c_m
        
        
        if calculateSimilarity == 'True':
            if 'p' not in locals():
                print 'Ciao!'
            
            
            print r['ds_tar'].shape
            print r['ds_src'].shape
            
            mahala_data = similarity_measure(r['ds_tar'], r['ds_src'], 
                                             r, p_value=p, method='correlation')
            r['mahalanobis_similarity'] = mahala_data
            #print tr_pred
        
            c_mat_mahala = ConfusionMatrix(predictions=mahala_data[0][mahala_data[1]].T[1], 
                                           targets=mahala_data[0][mahala_data[1]].T[0])
            c_mat_mahala.compute()
            r['confusion_mahala'] = c_mat_mahala
        
        else:
            r['mahalanobis_similarity'] = []
            r['confusion_mahala'] = 'Null'
            
        d_prime, beta, c, c_new = signal_detection_measures(pred, targets, map_list)
        r['d_prime'] = d_prime
        r['beta'] = beta
        r['c'] = c
        r['confusion_total'] = c_new
        
        '''
        d_prime_maha, c_new_maha = d_prime_statistics(tr_pred.T[1], tr_pred.T[0], map_list)
        r['d_prime_maha'] = d_prime_maha
        r['confusion_tot_maha'] = c_new_maha
        '''
        
        total_results[subj] = r
        
    conf_src['analysis_type'] = 'transfer_learning'
    conf_src['analysis_task'] = source
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)
    
    if (analysis_type=='group'):
        path = path[0]
    
    save_results(path, total_results, conf_src)
    
    return [total_results, r['ds_src'], r['ds_tar']]


def test_searchlight(path, subjects, conf_file, type, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    data_path = conf['data_path']
    for subj in subjects:
        
        ds = load_dataset(data_path, subj, type, **conf)
        ds = preprocess_dataset(ds, type, **conf)
        
        r = searchlight(ds, **kwargs)
        
        total_results[subj] = r
    
    conf['analysis_type'] = 'searchlight'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    save_results(path, total_results, conf)
    
    return total_results

def test_searchlight_cross_decoding(path, subjects, conf_file, type, **kwargs):
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
        if arg == 'radius':
            radius = kwargs[arg]
    
    
    debug.active += ["SLC"]
    
    ds_merged = get_merged_ds(path, subjects, conf_file, type, **kwargs)
    
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    cv = CrossValidation(clf, NFoldPartitioner(attr='task'))
    
    maps = []
    
    for ds in ds_merged:
                
        ds.targets[ds.targets == 'point'] = 'face'
        ds.targets[ds.targets == 'saccade'] = 'place'
        
        sl = sphere_searchlight(cv, radius, space = 'voxel_indices')
    
        sl_map = sl(ds)
    
        sl_map.samples *= -1
        sl_map.samples +=  1
    
        nif = map2nifti(sl_map, imghdr=ds.a.imghdr)
        
        maps.append(nif)
        
        
    datetime = get_time()
    analysis = 'cross_searchlight'
    mask = conf['mask_area']
    task = type
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    parent_dir = os.path.join(path, '0_results', new_dir)
    
    for s, map in zip(subjects, maps):
        name = s
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'
        map.to_filename(os.path.join(results_dir, fname))
        
    
    return maps
        
        



def test_group_mvpa(path, subjects, analysis,  conf_file, type='task', **kwargs):
    
       
    return 'no'


#####################################################################

def signal_detection_measures(predictions, targets, map_list):
    
    '''
    map_list =  two-element list, is what we expect the classifier did first element
                first element is the prediction
                second element is the label of second ds to be predicted
    '''
    
    mask_predictions = np.array(predictions) == map_list[0]
    mask_targets = np.array(targets) == map_list[1]
    
    new_predictions = np.array(predictions, dtype='|S20').copy()
    new_targets = np.array(targets, dtype='|S20').copy()
    
    new_label_1 = map_list[0]+'-'+map_list[1]
    new_label_2 = np.unique(new_predictions[~mask_predictions])[0]+'-'+ \
                    np.unique(new_targets[~mask_targets])[0]
    
    new_predictions[mask_predictions] = new_label_1
    new_targets[mask_targets] = new_label_1
    
    new_predictions[~mask_predictions] = new_label_2
    new_targets[~mask_targets] = new_label_2
    
    
    c_matrix = ConfusionMatrix(predictions=new_predictions, targets=new_targets)
    c_matrix.compute()
    
    from scipy.stats import norm
    
    hit_rate = c_matrix.stats['TP']/np.float_(c_matrix.stats['P'])
    false_rate = c_matrix.stats['FP']/np.float_(c_matrix.stats['N'])
    
    d_prime = norm.ppf(hit_rate) - norm.ppf(false_rate)
    
    beta = - d_prime[0] * 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate)) 
    
    c = - 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate))
    
       
    return d_prime[0], beta[0], c[0], c_matrix
    
def subjects_merged_ds(path, subjects, conf_file, task, **kwargs):
    
    
    conf = read_configuration(path, conf_file, task)
   
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    data_path = conf['data_path']
    
    i = 0

    print 'Merging subjects from '+data_path
    
    for subj in subjects:
        
        ds = load_dataset(data_path, subj, task, **conf)
        ds = preprocess_dataset(ds, task, **conf)
        
        if i == 0:
            ds_merged = ds.copy()
        else:
            ds_merged = vstack((ds_merged, ds))
            ds_merged.a.update(ds.a)
        i = i + 1
        
        del ds

    return ds_merged, ['group'], conf


def sources_merged_ds(path_list, subjects_list, conf_list, task, **kwargs):
    
    ds_list = []
    for path, subjects, conf in zip(path_list, subjects_list, conf_list):
        
        ds, _, conf_n = subjects_merged_ds(path, subjects, conf, task, **kwargs)
        
        ds_list.append(ds)
        
    
    ds_new = vstack(ds_list)
    ds_new.a.update(ds_list[0].a) 
    print 'Merging from different sources ended... '
    print 'The number of subjects merged are '+str(len(np.unique(ds_new.sa.name)))
    
    return ds_new, ['group'], conf_n

   
def get_merged_ds(path, subjects, conf_file, source='task', **kwargs):
    
    
    #Mettere source e target nel conf!
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    
    if source == 'saccade':
        target = 'face'
    else:
        if source == 'face':
            target = 'saccade'
    
    ds_merged_list = []
    conf_src = read_configuration(path, conf_file, source)
    conf_tar = read_configuration(path, conf_file, target)
    
    ##############################################    
    ##############################################
    ##    conf_src['label_included'] = 'all'    ##   
    ##    conf_src['label_dropped'] = 'none'    ##
    ##    conf_src['mean_samples'] = 'False'    ##
    ##############################################
    ##############################################
    
    for arg in kwargs:
        conf_src[arg] = kwargs[arg]
        conf_tar[arg] = kwargs[arg]
    
    data_path = conf_src['data_path']
    
    for subj in subjects:
        print '--------'
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_src.sa['task'] = [source for s in range(ds_src.samples.shape[0])]
        
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
        ds_tar.sa['task'] = [target for s in range(ds_tar.samples.shape[0])]
        
        ds_merged = vstack((ds_src, ds_tar))
        ds_merged.a.update(ds_src.a)
        
        ds_merged_list.append(ds_merged)
        '''
        methods = ['iso', 'pca', 'forest', 'embedding', 'mds']
        
        for m, i in zip(methods, range(len(methods))):
            plot_scatter_2d(ds_merged, method=m, fig_number=i+1)
        
        r = spatial(ds_merged, **conf_src)
        
    return r
    '''
    return ds_merged_list
    
def _group_transfer_learning(path, subjects, analysis,  conf_file, source='task', analysis_type='single', **kwargs):
    
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    
    if source == 'saccade':
        target = 'face'
    else:
        if source == 'face':
            target = 'saccade'
    
   
    ##############################################    
    ##############################################
    ##    conf_src['label_included'] = 'all'    ##   
    ##    conf_src['label_dropped'] = 'none'    ##
    ##    conf_src['mean_samples'] = 'False'    ##
    ##############################################
    ##############################################

    if analysis_type == 'group':
        
        if path.__class__ == conf_file.__class__ == list:  
            ds_src, s, conf_src = sources_merged_ds(path, subjects, conf_file, source, **kwargs)
            
            conf_src['permutations'] = 0
            
        else:
            print 'In group analysis path, subjects and conf_file must be lists: \
                    Check configuration file and/or parameters!!'
            return 0
    
    else:
        
        conf_src = read_configuration(path, conf_file, source)
        
    
    
    for arg in conf_src:
        if arg == 'map_list':
            map_list = conf_src[arg].split(',')
    
    
    r_group = spatial(ds_src, **conf_src)
    
    total_results = dict()
    total_results['group'] = r_group
    
    clf = r_group['classifier']
    
    for subj_, conf_, path_ in zip(subjects, conf_file, path):
        for subj in subj_:
            print '-----------'
            r = dict()
            if len(subj_) > 1:
                conf_tar = read_configuration(path_, conf_, target)
        
                for arg in kwargs:
                    
                    conf_tar[arg] = kwargs[arg]
            
            
                data_path = conf_tar['data_path']
                try:
                    ds_tar = load_dataset(data_path, subj, target, **conf_tar)
                except Exception, err:
                    print err
                    continue
    
            
            ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
    
            if conf_src['label_included'] == 'all' and \
               conf_src['label_dropped'] != 'fixation':
                    print 'Balancing dataset...'
                    ds_src = balance_dataset(ds_src, 'fixation')       
                    
             
            
            predictions = clf.predict(ds_tar)
            
            
            
            pred = np.array(predictions)
            targets = ds_tar.targets
            
            
            for arg in r_group.keys():
                r[arg] = copy.copy(r_group[arg])
            
            r['targets'] = targets
            r['predictions'] = predictions
            
            r['fclf'] = clf
            
            c_m = ConfusionMatrix(predictions=pred, targets=targets)
            c_m.compute()
            r['confusion_target'] = c_m
            print c_m
            
            tr_pred = similarity_measure_mahalanobis(ds_tar, ds_src, r)
            r['mahalanobis_similarity'] = tr_pred
            
            #print tr_pred
            
            c_mat_mahala = ConfusionMatrix(predictions=tr_pred.T[1], targets=tr_pred.T[0])
            c_mat_mahala.compute()
            r['confusion_mahala'] = c_mat_mahala
            
            d_prime, beta, c, c_new = signal_detection_measures(pred, targets, map_list)
            r['d_prime'] = d_prime
            print d_prime
            r['beta'] = beta
            r['c'] = c
            r['confusion_total'] = c_new
            
            '''
            d_prime_maha, c_new_maha = d_prime_statistics(tr_pred.T[1], tr_pred.T[0], map_list)
            r['d_prime_maha'] = d_prime_maha
            r['confusion_tot_maha'] = c_new_maha
            '''
            
            total_results[subj] = r
            
    group_k = set(total_results['group'].keys())
    subj_k = set(total_results[subj].keys())
    
    for k in subj_k.difference(group_k):
        total_results['group'][k] = copy.copy(total_results[subj][k])
    
    #total_results['group']['map'] = None   
    conf_src['analysis_type'] = 'transfer_learning'
    conf_src['analysis_task'] = source
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)
    
    if (analysis_type=='group'):
        path = path[0]
    
    save_results(path, total_results, conf_src)
    
    return total_results
    
