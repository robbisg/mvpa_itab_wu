#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import os
import cPickle as pickle
import nibabel as ni
import time

from mvpa2.suite import *

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.svm import SVC

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
        if (arg == 'img_dim'):
            img_dim = int(kwargs[arg])
                
    print 'Dataset preprocessing: Detrending and Z-Scoring...'
    if img_dim == 4:
        poly_detrend(ds, polyord = 1, chunks_attr = 'file');
    poly_detrend(ds, polyord = 1, chunks_attr = 'chunks');
    
    
                          
    if str(mean) == 'True':
        print 'Averaging samples...'
        avg_mapper = mean_group_sample(['events_number']) 
        ds = ds.get_mapped(avg_mapper)     

    if img_dim == 4:
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
        if arg == 'clf_type':
            clf_type = kwargs[arg]
        if arg == 'enable_results':
            enable_results = kwargs[arg].split(',')
    
    
    [fclf, cvte] = setup_classifier(**kwargs)
    
    
    print 'Cross validation is performing ...'
    error_ = cvte(ds)
    
    
    print cvte.ca.stats    
    #print error_.samples

    #Plot permutations
    #plot_cv_results(cvte, error_, 'Permutations analysis')
    dist_len = len(cvte.null_dist.dists())
    err_arr = np.zeros(dist_len)
    for i in range(dist_len):
        err_arr[i] = 1 - cvte.ca.stats.stats['ACC']
    
    total_p_value = np.mean(cvte.null_dist.p(err_arr))
    
    
    predictions_ds = fclf.predict(ds)
    
    '''
    If classifier didn't have sensitivity
    '''
    try:
        sensana = fclf.get_sensitivity_analyzer()
    except Exception, err:
        allowed_keys = ['map', 'sensitivities', 'stats', 
                        'mapper', 'classifier', 'ds', 
                        'p-value', 'p']
        
        allowed_results = [None, None, cvte.ca.stats, 
                           ds.a.mapper, fclf, ds, 
                           cvte.ca.null_prob.samples, total_p_value]
        
        results_dict = dict(zip(allowed_keys, allowed_results))
        results = dict()
        if not 'enable_results' in locals():
            enable_results = allowed_keys[:]
        for elem in enable_results:
            if elem in allowed_keys:
                results[elem] = results_dict[elem]
                
        return results
    
            
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

    classifier = fclf
    
    allowed_keys = ['map', 'sensitivities', 'stats', 
                    'mapper', 'classifier', 'ds', 
                    'p-value' , 'p']
    allowed_results = [l_maps, res_sens, cvte.ca.stats, 
                       ds.a.mapper, classifier, ds, 
                       cvte.ca.null_prob.samples, total_p_value]
    
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
    
    if 'duration' in locals():
        events = [e for e in events if e['duration'] >= duration]
    else:
        duration = np.min([ev['duration'] for ev in events])

    for e in events:
        e['onset'] += onset           
        e['duration'] = duration
        
    evds = eventrelated_dataset(ds, events = events) 
    
    [fclf, cvte] = setup_classifier(**kwargs)
    
    print 'Cross validation is performing ...'
    res = cvte(evds)
    
    print cvte.ca.stats 
    print cvte.ca.null_prob.samples
    
    dist_len = len(cvte.null_dist.dists())
    err_arr = np.zeros(dist_len)
    for i in range(dist_len):
        err_arr[i] = 1 - cvte.ca.stats.stats['ACC']
    
    total_p_value = np.mean(cvte.null_dist.p(err_arr))
    
    
    try:
        sensana = fclf.get_sensitivity_analyzer()
    except Exception, err:
        allowed_keys = ['map', 'sensitivities', 'stats', 
                        'mapper', 'classifier', 'ds', 
                        'p-value', 'p']
        
        allowed_results = [None, None, cvte.ca.stats, 
                           evds.a.mapper, fclf, evds, 
                           cvte.ca.null_prob.samples, total_p_value]
        
        results_dict = dict(zip(allowed_keys, allowed_results))
        results = dict()
        if not 'enable_results' in locals():
            enable_results = allowed_keys[:]
        for elem in enable_results:
            if elem in allowed_keys:
                results[elem] = results_dict[elem]
                
        return results
    
    res_sens = sensana(evds)
    
    sens_comb = res_sens.get_mapped(mean_sample())
    mean_map = map2nifti(evds, evds.a.mapper.reverse1(sens_comb))
        
    l_maps = []
    for m in res_sens:
        maps = ds.a.mapper.reverse1(m)
        nifti = map2nifti(evds, maps)
        l_maps.append(nifti)
    
    l_maps.append(mean_map)
    # Packing results    (to be sobstitute with a function)
    results = dict()
    if not 'enable_results' in locals():
        enable_results = ['map', 'sensitivities', 'stats', 
                          'mapper', 'classifier', 'ds', 
                          'p-value', 'p']
        
    allowed_keys = ['map', 'sensitivities', 'stats', 
                    'mapper', 'classifier', 'ds', 
                    'p-value', 'p']       
    
    allowed_results = [l_maps, res_sens, cvte.ca.stats, 
                       evds.a.mapper, fclf, evds, 
                       cvte.ca.null_prob.samples, total_p_value]
    
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
        enable_results = ['targets', 'classifier', 'map', 
                          'stats', 'sensitivities', 'mapper', 
                          'predictions','fclf', 'ds_src', 'ds_tar', 'p-value', 'p']
        
    allowed_keys = ['targets', 'classifier', 'map', 
                    'stats', 'sensitivities', 'mapper', 
                    'predictions','fclf', 'ds_src', 'ds_tar', 'p-value', 'p']
    
    
    if isinstance(classifier, FeatureSelectionClassifier):
        classifier_s = classifier.clf
    else:
        classifier_s = classifier
        
    allowed_results = [ds_tar.targets, classifier_s, src_result['map'], 
                       src_result['stats'], src_result['sensitivities'], 
                       src_result['mapper'], predictions, 
                       classifier, src_result['ds'], ds_tar, src_result['p-value'],
                       src_result['p'] ]
    
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
            if np.int(kwargs[arg]) == 0:
                cv_type =  np.float(kwargs[arg])
            else:
                cv_type = np.int(kwargs[arg])
        if arg == 'permutations':
            permutations = np.int(kwargs[arg])
        if arg == 'cv_attribute':
            attribute = kwargs[arg]
    
    cv_n = cv_type
            
    ################# Classifier #######################
    if clf_type == 'SVM':
        clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    elif clf_type == 'GNB':
        clf = GNB()
    elif clf_type == 'LDA':
        clf = LDA()
    elif clf_type == 'QDA':
        clf = QDA()
    elif clf_type == 'SMLR':
        clf = SMLR()
    elif clf_type == 'RbfSVM':
        sk_clf = SVC(gamma=0.1, C=1)
        clf = SKLLearnerAdapter(sk_clf, enable_ca=['probabilities'])
    elif clf_type == 'GP':
        clf = GPR()
    else:
        clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    
    ############## Feature Selection #########################    
    if f_sel == 'True':
        print 'Feature Selection selected.'
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                                FractionTailSelector(0.1, 
                                                                     mode = 'select', 
                                                                     tail = 'upper'))
        fclf = FeatureSelectionClassifier(clf, fsel)
    
    elif f_sel == 'Fixed':
        print 'Fixed Feature Selection selected.'
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                                FixedNElementTailSelector(50, 
                                                                     mode = 'select',
                                                                     tail = 'upper'))
        fclf = FeatureSelectionClassifier(clf, fsel)
    else:
        
        fclf = clf
    
    ######################### Permutations #############################
    
    if permutations != 0:
        if __debug__:
            debug.active += ["STATMC"]
        repeater = Repeater(count= permutations)
        permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)
        partitioner = NFoldPartitioner(cvtype=cv_n)
        null_cv = CrossValidation(
                                  clf,
                                  ChainNode([partitioner, permutator], 
                                            space=partitioner.get_space()),
                                  errorfx=mean_mismatch_error)
        
        distr_est = MCNullDist(repeater, tail='left', measure=null_cv,
                               enable_ca=['dist_samples'])
        #postproc = mean_sample()
    else:
        distr_est = None
        #postproc = None
    
    ##########################################################################    
    if cv_approach == 'n_fold':
        if cv_type != 0:
            cvte = CrossValidation(fclf, 
                                   NFoldPartitioner(cvtype = cv_type, attr = attribute), 
                                   #postproc = postproc,
                                   errorfx=mean_mismatch_error,
                                   null_dist=distr_est,
                                   enable_ca=['stats', 'repetition_results'])
        else:
            cvte = CrossValidation(fclf, 
                                   NFoldPartitioner(cvtype = 1, attr = attribute), 
                                   #postproc = postproc,
                                   errorfx=mean_mismatch_error,
                                   null_dist=distr_est,
                                   enable_ca=['stats', 'repetition_results'])
    else:
        cvte = CrossValidation(fclf, 
                               HalfPartitioner(attr = attribute), 
                               #postproc = postproc,
                               errorfx=mean_mismatch_error,
                               null_dist=distr_est,
                               enable_ca=['stats', 'repetition_results'])
        
    print 'Classifier set...'
    
    return [fclf, cvte]


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
