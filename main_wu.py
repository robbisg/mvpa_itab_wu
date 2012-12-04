import os
import cPickle as pickle
import nibabel as ni
import time

from mvpa2.suite import *
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.spatial.distance import *

import matplotlib.pyplot as plt
from io import *
from utils import *

class StoreResults(object):
        def __init__(self):
            self.storage = []
            
        def __call__(self, data, node, result):
            self.storage.append((node.measure.clf.ca.probabilities,
                                    node.measure.clf.ca.predictions)),
                                    
def balance_dataset(ds, label, sort=True, **kwargs):
    
    ################ To be changed ######################
    m_fixation = ds.targets == 'fixation'
    ev_fix = zip(ds.chunks[m_fixation], 4*((ds.sa.events_number[m_fixation]+2)/4 - 1 )+2)
    ####################################################
    
    ev_fix=np.array(ev_fix)
    ds.sa.events_number[m_fixation] = np.int_(ev_fix.T[1])
    arg_sort = np.argsort(ds.sa.events_number)
    events = find_events(chunks = ds[arg_sort].sa.chunks, targets = ds[arg_sort].sa.targets)
    # min duration
    min_duration = np.min( [e['duration'] for e in events])

    mask = False

    for ev in np.unique(ds.sa.events_number):
        mask_event = ds.sa.events_number == ev
        mask_event[np.nonzero(mask_event)[0][min_duration-1]+1:] = False
    
        mask = mask + mask_event
    
    if sort == True:
        arg_sort = np.argsort(ds[mask].sa.events_number)
        ds = ds[mask][arg_sort]
    else:
        ds = ds[mask]
    
    ds.a.events = find_events(targets = ds.targets, chunks = ds.chunks)
    
    return ds

def build_events_ds(ds, new_duration, **kwargs):
    """
    This function is used to convert a dataset in a event_related dataset. Used for
    transfer learning and clustering, thus a classifier has been trained on a 
    event related dataset and the prediction should be done on the same kind of the 
    dataset.
    
    Parameters    
    ----------
    
    ds : Dataset
        The dataset to be converted
    new_duration : integer
        Is the duration of the single event, if experiment events are of different
        length, it takes the events greater or equal to new_duration.
    kwarsg : dict
        win_number: is the number of window of one single event to be extracted,
        if it is not setted, it assumes the ratio between event duration and new_duration
        overlap:
        
    Returns
    -------
    
    Dataset:
        the event_related dataset
    """
    
    for arg in kwargs:
        if arg == 'number':
            win_number = kwargs[arg]
        if arg == 'overlap':
            overlap = kwargs[arg]

    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)
    labels = np.unique(ds.targets)
    current_duration = dict()
    for l in labels:
        d = [e['duration'] for e in events if e['targets'] == l]
        current_duration[l] = np.unique(d)[0]

    def calc_overlap(w, l, n):
        return w - np.floor((l - w)/(n - 1))
    
    def calc_win_number (w, l, o):
        return (l - w)/(w - o) + 1
    
    if 'overlap' not in locals():
        overlap = calc_overlap(new_duration, current_duration[l], win_number)
    else:
        if overlap >= new_duration:
            overlap = new_duration - 1
            
    if 'win_number' not in locals():
        #win_number = np.ceil(current_duration[l]/np.float(new_duration))
        win_number = calc_win_number(new_duration, current_duration[l], overlap)
        
    new_event_list = []
    
    for e in events:
        onset = e['onset']
        chunks = e['chunks']
        targets = e['targets']
        duration = e['duration']
        
        for i in np.arange(win_number):
            new_onset = onset + i * (new_duration - overlap)
            
            new_event = dict()
            new_event['onset'] = new_onset
            new_event['duration'] = new_duration
            new_event['targets'] = targets
            new_event['chunks'] = chunks
            
            new_event_list.append(new_event)
    
    
    print 'Building new event related dataset...'        
    evds = eventrelated_dataset(ds, events = new_event_list)
    
    return evds
    

def preprocess_dataset(ds, type, **kwargs):
    """
    Preprocess the dataset: detrending of single run and for chunks, the zscoring is also
    done by chunks and by run.
    
    Parameters
    ----------
    ds : Dataset
        The dataset to be preprocessed
    type : string
        The experiment to be processed
    kwargs : dict
        mean_samples - boolean : if samples should be averaged
        label_included - list : list of labels to be included in the dataset
        label_dropped - string : label to be dropped (rest, fixation)
        
    Returns
    -------
    Dataset
        the processed dataset
    
    
    """
    mean = False
    
    for arg in kwargs:
        if (arg == 'mean_samples'):
            mean = kwargs[arg]
        if (arg == 'label_included'):
            label_included = kwargs[arg].split(',')
        if (arg == 'label_dropped'):
            label_dropped = kwargs[arg] 
                
    print 'Dataset preprocessing: Detrending and Z-Scoring...'
    poly_detrend(ds, polyord = 1, chunks_attr = 'file');
    poly_detrend(ds, polyord = 1, chunks_attr = 'chunks');
    
    
                          
    if str(mean) == 'True':
        print 'Averaging samples...'
        avg_mapper = mean_group_sample(['events_number']) 
        ds = ds.get_mapped(avg_mapper)     
                
                

    
    zscore(ds, chunks_attr='file')
    zscore(ds)#, param_est=('targets', ['fixation']))
   
    if  label_dropped != 'none':
        ds = ds[ds.sa.targets != label_dropped]
    if  label_included != ['all']:
        ds = ds[np.array([l in label_included for l in ds.sa.targets],
                          dtype='bool')]
    
    ds.a.events = find_events(chunks = ds.sa.chunks, targets = ds.sa.targets)
    
    return ds


def spatial(ds, **kwargs):
#    gc.enable()
#    gc.set_debug(gc.DEBUG_LEAK)      
        
    for arg in kwargs:
        if arg == 'enable_results':
            enable_results = kwargs[arg]
    
    """
    [fclf, cvte, cv_storer] = setup_classifier(**kwargs)
    """   
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities', 'training_stats'])
    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.1, mode = 'select', tail = 'upper'))    
    fclf = FeatureSelectionClassifier(clf, fsel)
    cv_storage = StoreResults()
    cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = 1), callback=cv_storage,
                           enable_ca=['stats', 'repetition_results'],
                           )
    
  
    print 'Cross validation is performing ...'
    res = cvte(ds)
         
    print cvte.ca.stats    
    
    # Building sensitivities map.
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        sensana = fclf.get_sensitivity_analyzer()
    else:
        sensana = clf.get_sensitivity_analyzer()
        
    res_sens = sensana(ds)
    
    sens_comb = res_sens.get_mapped(mean_sample())
    mean_map = map2nifti(ds, ds.a.mapper.reverse1(sens_comb))

    
    l_maps = []
    for m in res_sens:
        maps = ds.a.mapper.reverse1(m)
        nifti = map2nifti(ds, maps)
        l_maps.append(nifti)
    
    
    l_maps.append(mean_map)
    # Packing results    (to be sobstitute with a function)
    results = dict()
    
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        classifier = fclf
    else:
        classifier = clf
    
    
    allowed_keys = ['map', 'sensitivities', 'stats', 'mapper', 'classifier',  'cv']
    allowed_results = [l_maps, res_sens, cvte.ca.stats, ds.a.mapper, classifier, cv_storage]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    if not 'enable_results' in locals():
        enable_results = allowed_keys[:]
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            print '******** '+elem+' result is not allowed! *********'

    
    return results



def searchlight(ds, **kwargs):
    
    if __debug__:
        debug.active += ["SLC"]
        
    radius = 3
    
    for arg in kwargs:
        if (arg == 'radius'):
            radius = kwargs[arg]
    
    """
    [fclf, cvte] = setup_classifier(**kwargs)
    """ 
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    cv = CrossValidation(clf, HalfPartitioner())
    
    
    sl = sphere_searchlight(cv, radius, space = 'voxel_indices')
    
    sl_map = sl(ds)
    
    sl_map.samples *= -1
    sl_map.samples +=  1
    
    nif = map2nifti(sl_map, imghdr=ds.a.imghdr)    
    
    #Results packing
    d_result = dict({'map': nif,
                     'radius': radius})
    
    return d_result


''' Fold average searchlight maps
for name in nameL:
    l = os.listdir(path+'/'+ name)
    lf = [s for s in l if s.find('searchlight') != -1 and s.find('mni') == -1]
    for sl in lf:
        o = ni.load(os.path.join(path, name, sl))
        nimg = np.mean(o.get_data(), axis=3)
        ni.save(ni.Nifti1Image(nimg, o.get_affine()), os.path.join(path, name, sl[:sl.rfind('_')]+'_avg.nii.gz'))
----------------------------------------------------------
i = 0
for list in lf_sl_total:
    if (i % 2) == 0:
        sum = np.zeros((48, 64, 48))
    else:
        sum = np.zeros((91,109, 91))
    i = i + 1
    for file in list:
        name = file.split('_')[0]
        img = ni.load(os.path.join(path, name, file))
        sum = sum + img.get_data()
    sum = sum / len(list)
    ni.save(ni.Nifti1Image(sum, img.get_affine()), os.path.join(path, 'group_avg'+file[file.find('_'):]))
 ----------------------------------------------------------   
    
    for m in lista_mask:
    refIn = os.path.join(path, 'andant','task', ref[:-4]+'.nii.gz')
    command = 'flirt '+ \
                      ' -in '+imgIn+ \
                      ' -ref '+refIn+ \
                      ' -init '+os.path.join(path,'iolpan','task',refL[6]) + \
                      ' -applyxfm -interp nearestneighbour' + \
                      ' -out '+ os.path.join(path,'1_single_ROIs','MNI',m[:-7]) +'_mni.nii.gz' 
    print command
    os.system(command)
'''
     
def spatiotemporal_(ds, **kwargs):
    onset = 0
    
    for arg in kwargs:
        if (arg == 'onset'):
            onset = kwargs[arg]
        if (arg == 'duration'):
            duration = kwargs[arg]
        if (arg == 'enable_results'):
            enable_results = kwargs[arg]
        
        
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)   
        
    if 'duration' in locals():
        events = [e for e in events if e['duration'] >= duration]
    else:
        duration = np.min([ev['duration'] for ev in events])

        
    evds = build_events_ds(ds, duration)
    
    """
    [fclf, cvte] = setup_classifier(**kwargs)
    """ 
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.1, mode = 'select', tail = 'upper'))    
    fclf = FeatureSelectionClassifier(clf, fsel)    
    
    
    print 'Cross validation...'
    cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
    train_err = cvte(evds)
    print cvte.ca.stats
    
    
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        sensana = fclf.get_sensitivity_analyzer()
    else:
        sensana = clf.get_sensitivity_analyzer()

    
    res_sens = sensana(evds)
    map = evds.a.mapper.reverse(res_sens.samples)
    

    # Packing results    (to be sobstitute with a function)
    results = dict()
    if not 'enable_results' in locals():
        enable_results = ['map', 'sensitivities', 'stats', 'mapper', 'classifier', 'cv']
        
    allowed_keys = ['map', 'sensitivities', 'stats', 'mapper', 'classifier', 'cv']
    
    
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        classifier = fclf
    else:
        classifier = clf
    
        
    allowed_results = [map, res_sens, cvte.ca.stats, evds.a.mapper, classifier, cvte]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            print '******** '+elem+' result is not allowed! *********'

    return results

def spatiotemporal(ds, **kwargs):
      
    onset = 0
    
    for arg in kwargs:
        if (arg == 'onset'):
            onset = kwargs[arg]
        if (arg == 'duration'):
            duration = kwargs[arg]
        if (arg == 'enable_results'):
            enable_results = kwargs[arg]
        
        
        
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)   
    
    #task_events = [e for e in events if e['targets'] in ['Vipassana','Samatha']]
    
    if 'duration' in locals():
        events = [e for e in events if e['duration'] >= duration]
    else:
        duration = np.min([ev['duration'] for ev in events])

    for e in events:
        e['onset'] += onset           
        e['duration'] = duration
        
    evds = eventrelated_dataset(ds, events = events) 
    
    """
    [fclf, cvte] = setup_classifier(**kwargs)
    """ 
    #clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities', 'training_stats'])
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.1, mode = 'select', tail = 'upper'))    
    fclf = FeatureSelectionClassifier(clf, fsel)    
    
    cv_storage = StoreResults()
    print 'Cross validation...'
    cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = 1), callback=cv_storage, 
                           enable_ca=['stats', 'repetition_results'])
    train_err = cvte(evds)
    print cvte.ca.stats
    
    
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        sensana = fclf.get_sensitivity_analyzer()
    else:
        sensana = clf.get_sensitivity_analyzer()

    
    res_sens = sensana(evds)
    sens_comb = res_sens.get_mapped(mean_sample())
    mean_map = map2nifti(evds, evds.a.mapper.reverse1(sens_comb))
        
    l_maps = []
    for m in res_sens:
        maps = evds.a.mapper.reverse1(m)
        nifti = map2nifti(evds, maps)
        l_maps.append(nifti)
    
    l_maps.append(mean_map)
    # Packing results    (to be sobstitute with a function)
    results = dict()
    if not 'enable_results' in locals():
        enable_results = ['map', 'sensitivities', 'stats', 'mapper', 'classifier', 'cv']
        
    allowed_keys = ['map', 'sensitivities', 'stats', 'mapper', 'classifier', 'cv']
    
    
    """ Not needed if setup_classifier"""
    if 'fclf' in locals():
        classifier = fclf
        
    else:
        classifier = clf   
        
    allowed_results = [l_maps, res_sens, cvte.ca.stats, evds.a.mapper, classifier, cv_storage]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            print '******** '+elem+' result is not allowed! *********'

    return results


def transfer_learning(ds_src, ds_tar, analysis, **kwargs):
    
    for arg in kwargs:
        if arg == 'enable_results':
            enable_results = kwargs[arg]
        if arg == 'duration':
            duration = kwargs[arg]  
        if arg == 'window_number':
            window_number = kwargs[arg]
    
    #src_result = analysis(ds_src, enable_results = ['classifier', 'map', 'stats', 'seni'], **kwargs)
    src_result = analysis(ds_src, **kwargs)
    
    classifier = src_result['classifier']
    
    if analysis.func_name == 'spatiotemporal':
        if 'duration' not in locals():
            duration = np.min([e['duration'] for e in ds_src.a.events])
            
        ds_tar = build_events_ds(ds_tar, duration, **kwargs)
        
    
    predictions = classifier.predict(ds_tar)
    
    ###########################################################
    #   Pack_results
    results = dict()
    #del enable_results
    if 'enable_results' not in locals():
        enable_results = ['targets', 'classifier', 'map', 'stats', 'sensitivities', 'mapper']
        
    allowed_keys = ['targets', 'classifier', 'map', 'stats', 'sensitivities', 'mapper']
    
    if isinstance(classifier, FeatureSelectionClassifier):
        classifier = classifier.clf

        
    allowed_results = [ds_tar.targets, classifier, src_result['map'], 
                       src_result['stats'], src_result['sensitivities'], src_result['mapper']]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            print '******** '+elem+' result is not allowed  ! *********'
    
    return results


def setup_classifier(**kwargs):
    
    '''
    Thinked!
    '''
    for arg in kwargs:
        if arg == 'clf_type':
            clf_type = kwargs[arg]
        if arg == 'fsel':
            f_sel = kwargs[arg]        
        if arg == 'cv_type':
            cv_approach = kwargs[arg]
        if arg == 'cv_folds':
            cv_type = kwargs[arg]    
    
    
    ################# Classifier #######################
    if clf_type == 'SVM':
        clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    elif clf_type == 'GNB':
        clf = GNB()
    elif clf_type == 'LDA':
        clf = LDA()
    else:
        clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    
    ############## Feature Selection #########################    
    if f_sel == 'True':
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                                FractionTailSelector(0.1, 
                                                                     mode = 'select', 
                                                                     tail = 'upper'))
        fclf = FeatureSelectionClassifier(clf, fsel)
    
    else:
        
        fclf = clf
    
    cv_storer = StoreResults()
    if cv_approach == 'n_fold':
        if cv_type in locals():
            cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = cv_type), callback=cv_storer,
                                   enable_ca=['stats', 'repetition_results'])
        else:
            cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = 1), callback=cv_storer,
                                   enable_ca=['stats', 'repetition_results'])
    else:
        cvte = CrossValidation(fclf, HalfPartitioner(), callback=cv_storer,
                                   enable_ca=['stats', 'repetition_results'])
        
    print 'Classifier set...'
    
    return [fclf, cvte, cv_storer]


def clustering (ds, n_clusters=6):
    
    data = ds.samples
    
    #clusters = inertia_clustering_analysis(ds, max_clusters = 13)
              
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    
    print 'Clustering with '+str(n_clusters)+' clusters...'
    cluster_label = kmeans.fit_predict(data)
    dist = squareform(pdist(data, 'euclidean'))
    
    ###########################################################
    #   Pack_results
    results = dict()

    if 'enable_results' not in locals():
        enable_results = ['clusters', 'dist']
        
    allowed_keys = ['clusters', 'dist']        
    allowed_results = [cluster_label, dist]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            print '******** '+elem+' result is not allowed! *********'
    
    return results
    
    
def clustering_analysis(ds_src, ds_tar, analysis, **kwargs):
    
    
    for arg in kwargs:
        if arg == 'duration':
            duration = kwargs[arg]
        if arg == 'mds':
            mds_flag = kwargs[arg]
            
        if arg == 'n_clusters':
            n_clusters = kwargs[arg]
            ########
    mds_flag = True
    
    
    r_trans = transfer_learning(ds_src, ds_tar, analysis, **kwargs)
    
    if analysis.func_name == 'spatiotemporal':
        if 'duration' not in locals():
            duration = np.min([e['duration'] for e in ds_src.a.events])
            
        ds_tar = build_events_ds(ds_tar, duration, **kwargs)
    
    r_clustering = dict()
    for label in np.unique(ds_tar.targets):
        mask = ds_tar.targets == label
        r_clustering[label] = clustering(ds_tar[mask], n_clusters=n_clusters)
        if mds_flag == True:
            mds = MDS(n_components=2, max_iter=2000,
                   eps=1e-9, random_state=seed,
                   n_jobs=1)
            print 'Multidimensional scaling is performing...'
            pos = mds.fit_transform(r_clustering[label]['dist'])
            r_clustering[label]['pos'] = pos
  
    #cluster_label = r_clustering['clusters']
    predictions = r_trans['classifier'].ca.predictions
    
    return dict({'clusters': r_clustering, 
                 'predictions': predictions, 
                 'targets': ds_tar.targets})
    

def inertia_clustering_analysis(ds, max_clusters = 13):
    
    inertia_val = np.array([])
    
    #max_clusters = 13#+2 = 15
    for i in np.arange(max_clusters)+2:
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
        kmeans.fit(ds.samples)
        inertia_val = np.append(inertia_val, kmeans.inertia_)
    
    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(inertia_val)
    plt.show()
    
    return inertia_val

              
def analyze (path, subjects, analysis, type, conf_file, **kwargs):
    
    
    configuration = read_configuration(path, conf_file, type)
    
    mask_area = ''
    mask_type = ''
    mask_space = ''
    
    for arg in kwargs:
        configuration[arg] = kwargs[arg]

    kwargs = configuration   
    #resFile = open(os.path.join(path, 'spatiotemporal_res_5.log'),'w')
    results = []

    for subj in subjects:
        ds = load_dataset(path, subj, type, **kwargs)
        if ds == 0:
            continue;
        else:
            ds = preprocess_dataset(ds, type, **kwargs)
            
            res = analysis(ds, **kwargs)

            mask = configuration['mask_atlas']+'_'+configuration['mask_area']
            
            results.append(dict({'name': subj,
                                'results': res,
                                'configuration': configuration}))
    
    #filename_pre = save_results(path, analysis, type, mask, results)

    return results


def watch_results(path, filename, write_map=True):
    results = pickle.load(open(os.path.join(path, 
                                        filename)))

       


def update_log (path, param, file_param):
    
    logFile = open(os.path.join(path, 'analysis.log'), 'a')
    
    
    logFile.write(file_param)
    logFile.write('\n---------------------\n')
    
    for el in param.iteritems():
        logFile.write(str(el))
        logFile.write('\n')
    
    logFile.write('---------------------\n')
    
    logFile.close()
    

def query_log (path, **kwargs):
    
    logFile = open(os.path.join(path, 'analysis.log'), 'a')
    
    return




################ Deprecated ###########################################

def transfer_learning_ (path, subjects, source, conf_src, conf_tar):
    #Source = dataset to train on the classifier
    """
    Deprecated
    """
    datetime = get_time()
    
    if source == 'rest':
        target = 'task'
    else:
        target = 'rest'
        
    res = []
    
    for subj in subjects:
        
        print ' ---------- Analyzing '+subj+' ----------------'
        
        print 'Loading source dataset: '+source
        ds_r = load_dataset(path, subj, source, **conf_src)
        
        print 'Loading target dataset.: '+target
        ds_t = load_dataset(path, subj, target, **conf_tar)
        
        if (ds_r == 0) or (ds_t == 0):
            continue;
        else:
            print 'Preprocessing...'
            ds_r = preprocess_dataset(ds_r, 'task', **conf_src)
            ds_t = preprocess_dataset(ds_t, 'rest', **conf_tar) 
            
            clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
            #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.1, mode = 'select', tail = 'upper')) 
        
            #fclf = FeatureSelectionClassifier(clf, fsel)  
            cvte = CrossValidation(clf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
            
            print 'Training...'
            train_err = cvte(ds_r)
            
            print 'Predicting...'
            predictions = clf.predict(ds_t)
            targets = ds_t.sa.targets
            values = clf.ca.estimates
            probabilities = clf.ca.probabilities
            
            
            print 'Saving results...'
            res.append(dict({'name':subj,
                        'predictions':predictions,
                        'targets':targets,
                        'values':values,
                        'training_error':train_err,
                        'stats': cvte.ca.stats,
                        'probabilities':probabilities,
                        'classifier': clf}))
            #pickle.dump(predictions, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_predict_no_fix.pyobj'), 'w'))
            #pickle.dump(targets, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_targets_no_fix.pyobj'), 'w'))
            #pickle.dump(values, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_values_no_fix.pyobj'), 'w'))
            #pickle.dump(values, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_values_no_fix.pyobj'), 'w'))
    
    area = conf_src['mask_area']      
    #pickle.dump(res, open(os.path.join(path, '0_results', datetime+'_res_'+source+'_'+area+'_transLearn_.pyobj'), 'w'))
    return res
    
def transfer_learning_fixation(path, subjects):


    for subj in subjects:
        
        print ' ---------- Analyzing '+subj+' ----------------'
        
        print 'Loading dataset...'
        ds_r = load_dataset(path, subj, 'task')
        ds_t = load_dataset(path, subj, 'task')
        
        if (ds_r == 0):
            continue;
        else:
            print 'Preprocessing...'
            ds_r = preprocess_dataset(ds_r, 'task')
            ds_t = preprocess_dataset(ds_t, 'rest') #Is rest because I need fixation volumes
            
            ds_t = ds_t[ds_t.sa.targets == 'fixation']
            
            clf = LinearCSVMC(C=1, probability=1)
            #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.1, mode = 'select', tail = 'upper')) 
        
            #fclf = FeatureSelectionClassifier(clf, fsel)  
            cvte = CrossValidation(clf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
            print 'Training...'
            train_err = cvte(ds_r)
            
            print 'Predicting...'
            predictions = clf.predict(ds_t)
            targets = ds_t.sa.targets
            values = clf.ca.estimates
            
            print 'Saving results...'
            source = 'fixation'
            pickle.dump(predictions, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_predict_.pyobj'), 'w'))
            pickle.dump(targets, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_targets_.pyobj'), 'w'))
            pickle.dump(values, open(os.path.join(path, '0_results', subj+'_src_'+source+'_transfLearn_values_.pyobj'), 'w'))


def clustering_analysis_(ds, classifier, clusters=6):
    """ Deprecated """
    
    data = ds.samples
    
    #clusters = inertia_clustering_analysis(ds, max_clusters = 13)
              
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
    
    cluster_label = kmeans.fit_predict(data)
    
    ################################################
    dist = squareform(pdist(data, 'euclidean'))
    
    mds = MDS(n_components=2, max_iter=3000,
                   eps=1e-9, random_state=seed,
                   n_jobs=1)
    pos = mds.fit_transform(dist)
    ###############################################
    
        
    f = plt.figure()
    a = f.add_subplot(111) 
    
    f_pr = plt.figure()
    a_pr = f_pr.add_subplot(111)
    ##################################################
    colors = cycle('bgrcmykbgrmykbgrcmykbgrcmyk')
    
    for cluster in np.unique(cluster_label):
        cl_mask = cluster_label == cluster
        c_pos = pos[cl_mask]
        col = colors.next()
        a.scatter(c_pos.T[0], c_pos.T[1], color = col)
        
        d_cl = ds[cl_mask]
        predictions = classifier.predict(d_cl)
        predictions = np.array(predictions)
        print 'Cluster n.: '+str(cluster)
    #####################################################
        for label in np.unique(predictions):
            p_mask = predictions == label
            labelled = np.count_nonzero(predictions == label)
            perc = np.float(labelled)/np.count_nonzero(cl_mask)
            a_pr.scatter(c_pos.T[0][p_mask], c_pos.T[1][p_mask], color = col, 
                         marker = markers[label])
            print label+' : '+str(perc*100)
    
    a.legend(np.unique(cluster_label))
    a_pr.legend(np.unique(cluster_label))
    plt.show()



if __name__ == '__main__':
    print 'Hello Guys'
    '''
    for p, t, v in zip(pred_files, target_files, val_files):
                predictions = pickle.load(open(os.path.join(path, '0_results', p), 'r'))
                values = pickle.load(open(os.path.join(path, '0_results', v), 'r'))
                target = pickle.load(open(os.path.join(path, '0_results', t), 'r'))
                
                if (p.find('src_task')==-1):
                    tar_label = 'RestPost'
                    pred_label = 'trained'
                else:
                    tar_label = 'RestPost'
                    pred_label = 'trained'
                
                mask_fix = np.array(target == tar_label)
                
                zipp = np.array(zip(target, predictions, values))
                
                filteredPre = zipp[mask_fix]
                
                pred = filteredPre.T[1]
                
                nPre = np.count_nonzero(np.array(pred == pred_label, dtype = 'int'))
                
                perc = float(nPre)/filteredPre.shape[0]
                
                print p[:p.find('transf')] + ' ' + str(perc)
        ------------------------------------------------------------------------------------------

           ----------------------------
             def test_var_kwargs(farg, **kwargs):
                print "formal arg:", farg
                radius = 3
                mask = 'total'
                for key in kwargs:
                    if (key == 'radius'):
                        radius = kwargs[key]
                    if (key == 'mask'):
                        mask = kwargs[key]  
                
                print mask + ' ' + str(radius)
                '''
