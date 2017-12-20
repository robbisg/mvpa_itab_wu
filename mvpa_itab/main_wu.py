#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

# pylint: disable=maybe-no-member, method-hidden
from sklearn.svm import SVC
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from mvpa_itab.io.base import *
from utils import *
from mvpa_itab.similarity.searchlight import *
from mvpa2.suite import eventrelated_dataset, find_events, debug
from mvpa2.suite import poly_detrend, zscore, mean_group_sample, mean_sample
from mvpa2.suite import LinearCSVMC, GNB, QDA, LDA, SMLR, GPR, SKLLearnerAdapter
from mvpa2.suite import map2nifti, mean_mismatch_error, sphere_searchlight
from mvpa2.suite import FractionTailSelector, FixedNElementTailSelector
from mvpa2.generators.partition import HalfPartitioner, NFoldPartitioner
from mvpa2.generators.resampling import Balancer
from mvpa2.measures.anova import OneWayAnova
from mvpa2.featsel.base import SensitivityBasedFeatureSelection
from mvpa2.suite import ChainNode, MCNullDist, Repeater, AttributePermutator
from mvpa2.suite import CrossValidation
from mvpa2.mappers.detrend import PolyDetrendMapper
import os
import cPickle as pickle
import nibabel as ni
from mvpa_itab import timewise
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
from mvpa2.clfs.meta import FeatureSelectionClassifier

logger = logging.getLogger(__name__)

class StoreResults(object):
    def __init__(self):
        self.storage = []
            
    def __call__(self, data, node, result):
        self.storage.append(node.measure.ca.predictions),


def balance_dataset_timewise(ds, label, sort=True, **kwargs):
    
    
    ################ To be changed ######################
    m_fixation = ds.targets == 'fixation'
    ev_fix = zip(ds.chunks[m_fixation], 
                 4*((ds.sa.events_number[m_fixation]+2)/4 - 1 )+2)
    ####################################################
    
    ev_fix=np.array(ev_fix)
    ds.sa.events_number[m_fixation] = np.int_(ev_fix.T[1])
    arg_sort = np.argsort(ds.sa.events_number)
    events = find_events(chunks = ds[arg_sort].sa.chunks, 
                         targets = ds[arg_sort].sa.targets)
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
        if arg == 'win_number':
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
    
    
    logger.info('Building new event related dataset...')
    evds = eventrelated_dataset(ds, events = new_event_list)
    
    return evds


def change_target(ds, target_name):
    
    ds.sa.targets = ds.sa[target_name]
    
    return ds



def slice_dataset(ds, selection_dict):
    """
    Select only portions of the dataset based on a dictionary
    The dictionary indicates the sample attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'accuracy': ['I'],
                        'frame':[1,2,3]
                        }
    
    """
    selection_mask = np.ones_like(ds.targets, dtype=np.bool)
    for key, values in selection_dict.iteritems():
        
        logger.info("Selected %s from %s attribute." %(str(values), key))
        
        ds_values = ds.sa[key].value
        condition_mask = np.zeros_like(ds_values, dtype=np.bool)
        
        for value in values:        
            condition_mask = np.logical_or(condition_mask, ds_values == value)
            
        selection_mask = np.logical_and(selection_mask, condition_mask)
        
    
    return ds[selection_mask]
        




def detrend_dataset(ds, type_, **kwargs):
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
    warnings.warn("Deprecated: This is going to be "+
                  "substituted with pipelines.", DeprecationWarning)
    
    warnings.warn("This function doesn't normalize features"+
                    " and/or samples. See normalize_dataset.")
    
    warnings.warn("This function doesn't select samples automatically."+
                  " Use slice_dataset to select samples from the dataset.")
    
    warnings.warn("This function doesn't select the target. "+
                  "Use change_target to use another attribute as target.")
    
    
    order = [1]
    
    for arg in kwargs:
        if arg == 'order':
            order = kwargs[arg]
            
    
    
    for o in order:
           
        logger.info('Dataset preprocessing: Detrending with polynomial of order %s...' %(str(o)))
        if len(np.unique(ds.sa['file'])) != 1:
            detr1 = PolyDetrendMapper(chunks_attr='file', polyord=o)
            ds = detr1.forward(ds)
            
        detr2 = PolyDetrendMapper(chunks_attr='chunks', polyord=o)
        ds = detr2.forward(ds)

    
    return ds



def balance_dataset(**kwargs):
    
    default_args = {
                    'amount':'equal', 
                    'attr':'targets', 
                    'count':10, 
                    'limit':None,
                    'apply_selection': True
                    }
    
    for arg in kwargs:
        if (arg.find('balancer') != -1):
            key = arg[arg.find('__')+2:]
            default_args[key] = kwargs[arg]
    
    balancer = Balancer(**default_args)
     
    return balancer
    


def chunk_dataset(ds, **kwargs):
    
    import collections
    import fractions
    
    chunk_number = None
    
    for arg in kwargs:
        if (arg == 'chunk_number'):
            chunk_number = kwargs[arg]
    
    
    n_targets = np.array([value for value in collections.Counter(ds.targets).values()]).min()
    
    if chunk_number == 'adaptive':
        n_chunks = np.max([fractions.gcd(n_targets, i) for i in np.arange(2, 10)])
        if n_chunks == 1:
            n_chunks = 4
    elif isinstance(chunk_number, int):
        n_chunks = int(chunk_number)
        
    if chunk_number != None:
        argsort = np.argsort(ds.targets)
        chunks = []
        for _ in ds.uniquetargets:
            chunk = np.linspace(0, n_chunks, n_targets, endpoint=False, dtype=np.int)
            chunks.append(chunk)
        
        
        ds.chunks[argsort] = np.hstack(chunks)   
    
    
    return ds



def normalize_dataset(ds, **kwargs):
    
    import collections
    import fractions
    
    mean = False
    normalization = 'feature'
    chunk_number = None
    
    for arg in kwargs:
        if (arg == 'mean_samples'):
            mean = kwargs[arg]
        if (arg == 'img_dim'):
            img_dim = int(kwargs[arg])
        if (arg == 'normalization'):
            normalization = str(kwargs[arg])
        if (arg == 'chunk_number'):
            chunk_number = kwargs[arg]
        
    n_targets = np.array([value for value in collections.Counter(ds.targets).values()]).min()
    
    if chunk_number == 'adaptive':
        n_chunks = np.max([fractions.gcd(n_targets, i) for i in np.arange(2, 10)])
        if n_chunks == 1:
            n_chunks = 4
    elif isinstance(chunk_number, int):
        n_chunks = int(chunk_number)
        
    if chunk_number != None:
        argsort = np.argsort(ds.targets)
        chunks = []
        for _ in ds.uniquetargets:
            chunk = np.linspace(0, n_chunks, n_targets, endpoint=False, dtype=np.int)
            chunks.append(chunk)
        
        
        ds.chunks[argsort] = np.hstack(chunks)
        
    
    if str(mean) == 'True':
        logger.info('Dataset preprocessing: Averaging samples...')
        avg_mapper = mean_group_sample(['event_num']) 
        ds = ds.get_mapped(avg_mapper)     
    
    
    if normalization == 'feature' or normalization == 'both':
        logger.info('Dataset preprocessing: Normalization feature-wise...')
        if img_dim == 4:
            zscore(ds, chunks_attr='file')
        zscore(ds)#, param_est=('targets', ['fixation']))
    
    
    if normalization == 'sample' or normalization == 'both':
        # Normalizing image-wise
        logger.info('Dataset preprocessing: Normalization sample-wise...')
        ds.samples -= np.mean(ds, axis=1)[:, None]
        ds.samples /= np.std(ds, axis=1)[:, None]
        
        ds.samples[np.isnan(ds.samples)] = 0
    
    
    # Find event related stuff
    ds.a.events = find_events(#event= ds.sa.event_num, 
                              chunks = ds.sa.chunks, 
                              targets = ds.sa.targets)
    
    return ds



def find_events_dataset(ds, **kwargs):
    
    ds.a.events = find_events(#event= ds.sa.event_num, 
                              chunks = ds.sa.chunks, 
                              targets = ds.sa.targets)
    
    return ds




def spatial(ds, **kwargs):
#    gc.enable()
#    gc.set_debug(gc.DEBUG_LEAK)      
    
    cvte = None
    permutations = 0
    
    for arg in kwargs:
        if arg == 'clf_type':
            clf_type = kwargs[arg]
        if arg == 'enable_results':
            enable_results = kwargs[arg].split(',')
        if arg == 'permutations':
            permutations = int(kwargs[arg])
        if arg == 'cvte':
            cvte = kwargs[arg]
            fclf = cvte.learner # Send crossvalidation object
    
    if cvte == None:
        [fclf, cvte] = setup_classifier(**kwargs)
    
    
    logger.info('Cross validation is performing ...')
    error_ = cvte(ds)
    
    
    logger.debug(cvte.ca.stats)
    #print error_.samples

    #Plot permutations
    #plot_cv_results(cvte, error_, 'Permutations analysis')
    
    if permutations != 0:
        dist_len = len(cvte.null_dist.dists())
        err_arr = np.zeros(dist_len)
        for i in range(dist_len):
            err_arr[i] = 1 - cvte.ca.stats.stats['ACC']
    
        total_p_value = np.mean(cvte.null_dist.p(err_arr))
        p_value = cvte.ca.null_prob.samples
        
    else:
        p_value = np.array([0, 0])
        total_p_value = 0
    
    predictions_ds = fclf.predict(ds)
    

    # If classifier didn't have sensitivity
    try:
        sensana = fclf.get_sensitivity_analyzer()
    except Exception, err:
        allowed_keys = ['map', 'sensitivities', 'stats', 
                        'mapper', 'classifier', 'ds_src', 
                        'perm_pvalue', 'p']
        
        allowed_results = [None, None, cvte.ca.stats, 
                           ds.a.mapper, fclf, ds, 
                           p_value, total_p_value]
        
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
                    'mapper', 'classifier', 'ds_src', 
                    'pvalue' , 'p']
    allowed_results = [l_maps, res_sens, cvte.ca.stats, 
                       ds.a.mapper, classifier, ds, 
                       p_value, total_p_value]
    
    results_dict = dict(zip(allowed_keys, allowed_results))
    
    if not 'enable_results' in locals():
        enable_results = allowed_keys[:]
    
    for elem in enable_results:
        
        if elem in allowed_keys:
            results[elem] = results_dict[elem]
        else:
            logger.error('******** '+elem+' result is not allowed! *********')

    
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
    cv = CrossValidation(clf, HalfPartitioner(attr='chunks'))
    
    
    sl = sphere_searchlight(cv, radius, space = 'voxel_indices')
    
    #sl = Searchlight(MahalanobisMeasure, queryengine, add_center_fa, results_postproc_fx, results_backend, results_fx, tmp_prefix, nblocks)
    #sl = sphere_searchlight(MahalanobisMeasure(), 3, space= 'voxel_indices')
    sl_map = sl(ds)
    
    sl_map.samples *= -1
    sl_map.samples +=  1

    nif = map2nifti(sl_map, imghdr=ds.a.imghdr)
    nif.set_qform(ds.a.imgaffine)  
    
    #Results packing
    d_result = dict({'map': nif,
                     'radius': radius})
    
    return d_result


def spatiotemporal(ds, **kwargs):
      
    onset = 0
    
    for arg in kwargs:
        if (arg == 'onset'):
            onset = kwargs[arg]
        if (arg == 'duration'):
            duration = kwargs[arg]
        if (arg == 'enable_results'):
            enable_results = kwargs[arg]
        if (arg == 'permutations'):
            permutations = int(kwargs[arg])
       
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
    
    logger.info('Cross validation is performing ...')
    res = cvte(evds)
    
    print cvte.ca.stats 
    
    
    if permutations != 0:
        print cvte.ca.null_prob.samples
        dist_len = len(cvte.null_dist.dists())
        err_arr = np.zeros(dist_len)
        for i in range(dist_len):
            err_arr[i] = 1 - cvte.ca.stats.stats['ACC']
    
        total_p_value = np.mean(cvte.null_dist.p(err_arr))
        p_value = cvte.ca.null_prob.samples
    else:
        total_p_value = 0.
        p_value = np.array([0,0])
    
    
    try:
        sensana = fclf.get_sensitivity_analyzer()
        res_sens = sensana(evds)
    except Exception, err:
        allowed_keys = ['map', 'sensitivities', 'stats', 
                        'mapper', 'classifier', 'ds', 
                        'perm_pvalue', 'p']
        
        allowed_results = [None, None, cvte.ca.stats, 
                           evds.a.mapper, fclf, evds, 
                           p_value, total_p_value]
        
        results_dict = dict(zip(allowed_keys, allowed_results))
        results = dict()
        if not 'enable_results' in locals():
            enable_results = allowed_keys[:]
        for elem in enable_results:
            if elem in allowed_keys:
                results[elem] = results_dict[elem]
                
        return results
    
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
                          'pvalue', 'p']

    allowed_keys = ['map', 'sensitivities', 'stats',
                    'mapper', 'classifier', 'ds',
                    'pvalue', 'p']

    allowed_results = [l_maps, res_sens, cvte.ca.stats, 
                       evds.a.mapper, fclf, evds, 
                       p_value, total_p_value]

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
    if isinstance(classifier, FeatureSelectionClassifier):
        classifier_s = classifier.clf
    else:
        classifier_s = classifier
    
    results = dict()
        
    allowed_keys = ['targets', 
                    'classifier', 
                    'map',
                    'stats', 
                    'sensitivities', 
                    'mapper',
                    'predictions',
                    'fclf', 
                    'ds_src', 
                    'ds_tar', 
                    'perm_pvalue', 
                    'p']
    
          
   
    allowed_results = [ds_tar.targets, 
                       classifier_s, 
                       src_result['map'], 
                       src_result['stats'], 
                       src_result['sensitivities'], 
                       src_result['mapper'], 
                       predictions, 
                       classifier, 
                       src_result['ds'], 
                       ds_tar, 
                       src_result['pvalue'],
                       src_result['p'] ]

    
    results_dict = dict(zip(allowed_keys, allowed_results))

    for elem in allowed_keys:
        results[elem] = results_dict[elem]

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
                cv_type = np.float(kwargs[arg])
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
        logger.info('Feature Selection selected.')
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                                FractionTailSelector(0.05,
                                                                     mode='select',
                                                                     tail='upper'))
        fclf = FeatureSelectionClassifier(clf, fsel)

    elif f_sel == 'Fixed':
        logger.info('Fixed Feature Selection selected.')
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  
                                                FixedNElementTailSelector(100,
                                                                     mode='select',
                                                                     tail='upper'))
        fclf = FeatureSelectionClassifier(clf, fsel)
        
    elif f_sel == 'PCA':
        from mvpa2.mappers.skl_adaptor import SKLTransformer
        from sklearn.decomposition import PCA
        logger.info('Fixed Feature Selection selected.')
        fsel = SKLTransformer(PCA(n_components=45))
        
        fclf = FeatureSelectionClassifier(clf, fsel)
    else:

        fclf = clf

    ######################### Permutations #############################

    if permutations != 0:
        if __debug__:
            debug.active += ["STATMC"]
        repeater = Repeater(count=permutations)
        permutator = AttributePermutator('targets', limit={'partitions': 1}, 
                                         count=1)
        partitioner = NFoldPartitioner(cvtype=cv_n, attr=attribute)
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

    ########################################################
    if cv_approach == 'n_fold':
        if cv_type != 0:
            splitter_used = NFoldPartitioner(cvtype=cv_type, attr=attribute)
        else:
            splitter_used = NFoldPartitioner(cvtype=1, attr=attribute)
    else:
        splitter_used = HalfPartitioner(attr=attribute)
        
    
    chain_splitter = ChainNode([splitter_used, 
                                Balancer(attr='targets',
                                         count=1,
                                         limit='partitions',
                                         apply_selection=True)],
                               space='partitions')

    #############################################################
    if distr_est == None:
        cvte = CrossValidation(fclf,
                               chain_splitter,
                               enable_ca=['stats', 'repetition_results'])
    else:
        cvte = CrossValidation(fclf,
                               chain_splitter,
                               errorfx=mean_mismatch_error,
                               null_dist=distr_est,
                               enable_ca=['stats', 'repetition_results'])

    logger.info('Classifier set...')

    return [fclf, cvte]




def clustering(ds, n_clusters=6):

    from sklearn.manifold import MDS
    from sklearn.cluster import KMeans

    data = ds.samples

    #clusters = inertia_clustering_analysis(ds, max_clusters = 13)

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)

    logger.info('Clustering with ' + str(n_clusters) + ' clusters...')
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
            logger.error('******** ' + elem + ' result is not allowed! *********')

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
                   eps=1e-9, n_jobs=1)
            logger.info('Multidimensional scaling is performing...')
            pos = mds.fit_transform(r_clustering[label]['dist'])
            r_clustering[label]['pos'] = pos

    #cluster_label = r_clustering['clusters']
    predictions = r_trans['classifier'].ca.predictions

    return dict({'clusters': r_clustering, 
                 'predictions': predictions, 
                 'targets': ds_tar.targets})



def inertia_clustering_analysis(ds, max_clusters=13):

    inertia_val = np.array([])

    #max_clusters = 13#+2 = 15
    for i in np.arange(max_clusters)+2:
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
        kmeans.transform(ds.samples)
        inertia_val = np.append(inertia_val, kmeans.inertia_)

    f = plt.figure()
    a = f.add_subplot(111)
    a.plot(inertia_val)
    plt.show()

    return inertia_val


def analyze(path, subjects, analysis, model, conf_file, **kwargs):

    configuration = read_configuration(path, conf_file, model)

    mask_area = ''
    mask_type = ''
    mask_space = ''

    for arg in kwargs:
        configuration[arg] = kwargs[arg]

    kwargs = configuration
    #resFile = open(os.path.join(path, 'spatiotemporal_res_5.log'),'w')
    results = []

    for subj in subjects:
        ds = load_dataset(path, subj, model, **kwargs)
        if ds == 0:
            continue
        else:
            ds = detrend_dataset(ds, model, **kwargs)
            
            res = analysis(ds, **kwargs)

            mask = configuration['mask_atlas'] + '_' + configuration['mask_area']

            results.append(dict({'name': subj,
                                'results': res,
                                'configuration': configuration}))

    #filename_pre = save_results(path, analysis, type, mask, results)

    return results
