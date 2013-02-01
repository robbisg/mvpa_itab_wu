from main_wu import *
from io import *
from mvpa2.clfs.transerror import ConfusionMatrix
import os
    
    
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
        
        r = spatiotemporal(ds, **kwargs)
        
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
            ds_src = load_dataset(data_apath, subj, source, **conf_src)
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


def test_transfer_learning(path, subjects, analysis,  conf_file, source='task', **kwargs):
    
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    if source == 'lip':
        target = 'ffa'
    else:
        if source == 'ffa':
            target = 'lip'
    
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
    total_results = dict()
    
    for subj in subjects:
        print '--------'
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
                print 'Balancing dataset...'
                ds_src = balance_dataset(ds_src, 'fixation')        
        
        r = transfer_learning(ds_src, ds_tar, analysis, **conf_src)
        
        pred = np.array(r['classifier'].ca.predictions)
        targets = r['targets']
        
        c_m = ConfusionMatrix(predictions=pred, targets=targets)
        c_m.compute()
        r['confusion_target'] = c_m
        
        total_results[subj] = r
        
    conf_src['analysis_type'] = 'transfer_learning'
    conf_src['analysis_task'] = source
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)

    save_results(path, total_results, conf_src)
    
    return total_results


def test_searchlight(path, subjects, conf_file, type, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    
    for subj in subjects:
        
        ds = load_dataset(path, subj, type, **conf)
        ds = preprocess_dataset(ds, type, **conf)
        
        r = searchlight(ds, **kwargs)
        
        total_results[subj] = r
    
    conf['analysis_type'] = 'searchlight'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    save_results(path, total_results, conf)
    
    return total_results
#####################################################################
    