#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

from main_wu import *
from io import *
from io.base import *
from mvpa2.clfs.transerror import ConfusionMatrix
import os
import copy
from mvpa_itab.similarity.cross_decoding import *
from mvpa2.suite import vstack, Sphere, Searchlight, IndexQueryEngine, Splitter
import mvpa_itab.results as rs

logger = logging.getLogger(__name__)
    
def test_spatiotemporal(path, subjects, conf_file, type_, **kwargs):
    
    conf = read_configuration(path, conf_file, type_)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
        if arg == 'balance':
            balance = kwargs[arg]
    
    total_results = dict()
    
    data_path = conf['data_path']
    conf['analysis_type'] = 'spatiotemporal'
    conf['analysis_task'] = type_
    
    for subj in subjects:
        try:
            ds = load_dataset(data_path, subj, type_, **conf)
        except Exception, err:
            print err
            continue
        ds = preprocess_dataset(ds, type_, **conf)
        
        if 'balance' in locals() and balance == True:
            if conf['label_included'] == 'all' and \
                conf['label_dropped'] == 'none':
                ds = balance_dataset(ds, 'fixation')
        
        r = spatiotemporal(ds, **conf)
        
        total_results[subj] = r
    

    conf['classes'] = np.unique(ds.targets)
    
    save_results(path, total_results, conf)
    
    return total_results

def test_spatial(path, subjects, conf_file, type_, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type_)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    
    data_path = conf['data_path']
        
    conf['analysis_type'] = 'spatial'
    conf['analysis_task'] = type_
    
    result = rs.DecodingResults(path, conf)
    
    for subj in subjects:
        print '------'
        try:
            ds = load_dataset(data_path, subj, type_, **conf)
        except Exception, err:
            print err
            continue
        
        ds = preprocess_dataset(ds, type_, **conf)
        
        r = spatial(ds, **conf)
        total_results[subj] = r
        
        subj_result = rs.DecodingSubjectResult(subj, r)
        
        result.aggregate(subj_result)
        
    #result.save()
    result.summarize()

    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    #save_results(path, total_results, conf)
    
    return total_results, subj_result


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
    
    
    conf_src['analysis_type'] = 'transfer_learning'
    conf_src['analysis_task'] = source
    conf_src['analysis_func'] = analysis.func_name
    
    
    for arg in conf_src:
        if arg == 'map_list':
            map_list = conf_src[arg].split(',')
        if arg == 'p_dist':
            p = float(conf_src[arg])
            print p
    
    
    total_results = dict()
    
    
    
    
    summarizers = [rs.CrossDecodingSummarizer(),
                   rs.SimilaritySummarizer(),
                   rs.DecodingSummarizer(),
                   rs.SignalDetectionSummarizer(),
                   ]
    
    savers = [rs.CrossDecodingSaver(),
                   rs.SimilaritySaver(),
                   rs.DecodingSaver(),
                   rs.SignalDetectionSaver(),
                   ]
    
    collection = rs.ResultsCollection(conf_src, path, summarizers)
    
    
    for subj in subjects:
        print '-------------------'
        
        if (len(subjects) > 1) or (subj != 'group'):
            try:
                ds_src = load_dataset(data_path, subj, source, **conf_src)
                ds_tar = load_dataset(data_path, subj, target, **conf_tar)
            except Exception, err:
                print err
                continue
         
        # Evaluate if is correct to do further normalization after merging two ds. 
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
        
        if conf_src['label_included'] == 'all' and \
           conf_src['label_dropped'] != 'fixation':
                print 'Balancing dataset...'
                ds_src = balance_dataset(ds_src, 'fixation')        
        
        # Make cross-decoding
        r = transfer_learning(ds_src, ds_tar, analysis, **conf_src)
        
        
        
        
        # Now we have cross-decoding results we could process it
        pred = np.array(r['classifier'].ca.predictions)

        targets = r['targets']
        
        c_m = ConfusionMatrix(predictions=pred, targets=targets)
        c_m.compute()
        r['confusion_target'] = c_m
        
        c_new = cross_decoding_confusion(pred, targets, map_list)
        r['confusion_total'] = c_new
        
        print c_new
        
        # Similarity Analysis
        if calculateSimilarity == 'True':
            if 'p' not in locals():
                print 'Ciao!'

            
            mahala_data = similarity_measure(r['ds_tar'], r['ds_src'], 
                                             r, p_value=p, method='mahalanobis')
            
            #r['mahalanobis_similarity'] = mahala_data
            for k_,v_ in mahala_data.items():
                r[k_] = v_
            r['confusion_mahala'] = mahala_data['confusion_mahalanobis']
        
        else:
            #r['mahalanobis_similarity'] = []
            r['confusion_mahala'] = 'Null'
        
        # Signal Detection Theory Analysis
        sdt_res = signal_detection_measures(c_new)
        
        for k_,v_ in sdt_res.items():
            r[k_] = v_
            
            '''
            Same code of:
        
            r['d_prime'] = d_prime
            r['beta'] = beta
            r['c'] = c
            '''
        
        total_results[subj] = r
        subj_result = rs.SubjectResult(subj, r, savers=savers)
        
        collection.add(subj_result)
        # Save each subject
    collection.summarize()   
    
    conf_src['classes'] = np.unique(ds_src.targets)    

    
    if (analysis_type=='group'):
        path = path[0]
    
    #save_results(path, total_results, conf_src)
    
    return [total_results, r['ds_src'], r['ds_tar'], mahala_data]


def test_searchlight(path, subjects, conf_file, type_, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type_)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    
    conf['analysis_type'] = 'searchlight'
    conf['analysis_task'] = type_
    
    total_results = dict()
    data_path = conf['data_path']
    
    #
    summarizers = [rs.SearchlightSummarizer()]
    savers = [rs.SearchlightSaver()]
    result = rs.ResultsCollection(conf, path, summarizers)
    #
    
    for subj in subjects:
        
        ds = load_dataset(data_path, subj, type_, **conf)
        ds = preprocess_dataset(ds, type_, **conf)
        
        r = searchlight(ds, **kwargs)
        
        subj_result = rs.SubjectResult(subj, r, savers)
        total_results[subj] = r
        
        result.add(subj_result)
    
    result.summarize()
    

    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    #save_results(path, total_results, conf)
    
    return result, r, subj_result

def test_searchlight_similarity(path,
                                subjects, 
                                conf_file, 
                                type, 
                                dim=4,
                                measure=CorrelationMeasure(),
                                partitioner=TargetCombinationPartitioner(attr='targets'),
                                **kwargs):
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
        if arg == 'radius':
            radius = kwargs[arg]
        if arg == 'duration':
            duration = kwargs[arg]
            
    #########################################################
    datetime = get_time()
    analysis = 'searchlight_measure'
    mask = conf['mask_area']
    task = type
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    parent_dir = os.path.join(path, '0_results', new_dir)       
    ##########################################################
    
    debug.active += ["SLC"]
    
    ds_merged = get_merged_ds(path, subjects, conf_file, type, dim=dim, **kwargs)
    
    cv = CrossValidation(measure, 
                         partitioner,
                         splitter=Splitter(attr='partitions', attr_values=(3,2)),
                         errorfx=None
                         )
    
    maps = []
    kwa = dict(voxel_indices=Sphere(radius), 
               event_offsetidx=Sphere(duration))
    queryengine = IndexQueryEngine(**kwa)
        
    for s, ds in zip(subjects, ds_merged):
        
        print ds.samples.shape
        
        voxel_num = ds.samples.shape[1]/duration
        ids = np.arange(voxel_num)        
        sl = Searchlight(cv, queryengine=queryengine, roi_ids=ids)
    
        map = sl(ds)
            
        maps.append(map)

        name = s
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'        
        map_ = ds.a.mapper[:2].reverse(map)
        map_ = np.rollaxis(map_.samples, 0, 4)
        img = ni.Nifti1Image(map_, ds.a.imghdr.get_base_affine())
        ni.save(img, os.path.join(results_dir, fname))
    
    return maps


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

def cross_decoding_confusion(predictions, targets, map_list):
    '''
    map_list =  two-element list, is what we expect the classifier did when it cross-decodes
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
    
    return c_matrix

def signal_detection_measures(confusion_):
    
    result = dict()

    from scipy.stats import norm
    
    hit_rate = confusion_.stats['TP']/np.float_(confusion_.stats['P'])
    false_rate = confusion_.stats['FP']/np.float_(confusion_.stats['N'])
    
    d_prime = norm.ppf(hit_rate) - norm.ppf(false_rate)
    
    beta = - d_prime[0] * 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate)) 
    
    c = - 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate))
    
    result['beta'] = beta[0]
    result['d_prime'] = d_prime[0]
    result['c'] = c[0]
    
    return result
    
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

   
def get_merged_ds(path, subjects, conf_file, source='task', dim=3, **kwargs):
    
    
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
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 

        if dim == 4:    
            duration = np.min([e['duration'] for e in ds_src.a.events])      
            ds_tar = build_events_ds(ds_tar, duration, overlap=duration-1)
            ds_src = load_spatiotemporal_dataset(ds_src, duration=duration)
        
        print ds_src.samples.shape
        print ds_tar.samples.shape 
        
        ds_src.sa['task'] = [source for s in range(ds_src.samples.shape[0])]
        ds_tar.sa['task'] = [target for s in range(ds_tar.samples.shape[0])]
        
        ds_merged = vstack((ds_src, ds_tar))
        ds_merged.a.update(ds_src.a)
        
        print ds_merged.sa.task
        
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
    
