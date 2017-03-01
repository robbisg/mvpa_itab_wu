############## Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import preprocess_dataset
from mvpa_itab.timewise import AverageLinearCSVM, ErrorPerTrial, StoreResults
from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error
import nibabel as ni
import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer
import mvpa_itab.results as rs
from mvpa_itab.wrapper.sklearn import SKLCrossValidation
from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import SVC
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.suite import debug, sphere_searchlight
from mvpa2.suite import *

from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from numpy.random.mtrand import permutation
from mvpa_itab.test_wu import subjects_merged_ds
from mvpa_itab.similarity.partitioner import MemoryGroupSubjectPartitioner
#path = '/media/robbis/DATA/fmri/memory/'


import logging
logger = logging.getLogger(__name__)

def get_partitioner(split_attr='group_split'):
    
    if split_attr == 'group_split':
    
        splitrule = [
                    (['3','4'],['1'],['2']),
                    (['3','4'],['2'],['1']),
                    (['1'],['2'],['3','4']),
                    (['2'],['1'],['3','4']),
                    
                    (['1','2'],['3'],['4']),
                    (['1','2'],['4'],['3']),
                    (['3'],['4'],['1','2']),
                    (['4'],['3'],['1','2']),
                    ]
        partitioner = CustomPartitioner(splitrule=splitrule,
                                        attr=split_attr                                                
                                        )
                    
    elif split_attr == 'subject':
        
        partitioner = MemoryGroupSubjectPartitioner(group_attr='group_split', 
                                                    subject_attr=split_attr,
                                                    attr=split_attr)
                
    return partitioner


def experiment_conf(task_, ev):
    
    conf = dict()
    # label managing
    if task_ == 'memory':
        conf['field'] = 'stim'
        conf['label_dropped'] = 'F0'
        conf['label_included'] = 'N'+ev+','+'O'+ev
        conf['count'] = 1
    else: # decision
        conf['field'] = 'decision'
        conf['label_dropped'] = 'FIX0'
        conf['label_included'] = 'NEW'+ev+','+'OLD'+ev
        conf['count'] = 1
        
    return conf


def analysis(**kwargs):
    
    default_conf = {'path':'/home/robbis/fmri/memory/',
                    'data_type': 'BETA_MVPA',
                    'n_folds':3,
                    'evidences':[1,3,5],
                    'tasks':['memory', 'decision'],
                    'split_attr':'subject',
                    'mask_area':'intersect',
                    'normalization':'sample'
                        
                    }
    
    
    default_conf.update(kwargs)
    path = default_conf['path']
    
    if __debug__:
        debug.active += ["SLC"]
        
    evidences = default_conf['evidences']
    tasks = default_conf['tasks']
    
    conf = read_configuration(path, 'memory.conf', 'BETA_MVPA')
    
    conf['analysis_type'] = 'searchlight'
    conf['analysis_task'] = 'memory'
    
    summarizers = [rs.SearchlightSummarizer()]
    savers = [rs.SearchlightSaver()]
    collection = rs.ResultsCollection(conf, path, summarizers)

    data_type = default_conf['data_type']
    
    conf = read_configuration(path, 'memory.conf', data_type)
    data_path = conf['data_path']
    
    #ds_original = load_dataset(data_path, subj, data_type, **conf)
    ds_original, _, _ = subjects_merged_ds(path, 
                                               None, 
                                               'memory.conf', 
                                               'BETA_MVPA',
                                               subject_file=os.path.join(path, 'subjects.csv'),
                                               normalization=default_conf['normalization'],
                                               mask_area=default_conf['mask_area'])

    res = [] 
    result_dict = dict()
    
    partitioner = get_partitioner(default_conf['split_attr'])

    for task_ in tasks:
        for ev in evidences:
            
            print '---------------------------------'
            
            ev = str(ev)
            
            ds = ds_original.copy()
            
            ex_conf = experiment_conf(task_, ev)
            field_ = ex_conf.pop('field')
            count_ = 1
            conf.update(ex_conf)
            

            ds.targets = np.core.defchararray.add(np.array(ds.sa[field_].value, dtype=np.str), 
                                                  np.array(ds.sa.evidence, dtype= np.str))

            
            ds = preprocess_dataset(ds, data_type, **conf)
    
            balanc = Balancer(count=count_, 
                              apply_selection=True, 
                              limit='group_split')
            
            gen = balanc.generate(ds)
            
            cv_storage = StoreResults()

            maps = []
            
            clf = LinearCSVMC(C=1)
            
            for i, ds_ in enumerate(gen):
                #print ds_.summary(sstats=False)
                #Avoid balancing!
                #ds_ = ds
                
                # This is used for the sklearn crossvalidation
                y = np.zeros_like(ds_.targets, dtype=np.int_)
                y[ds_.targets == ds_.uniquetargets[0]] = 1
                
                # We needs to modify the chunks in order to use sklearn
                ds_.chunks = np.arange(len(ds_.chunks))
                
                #For each splitrule re-run searchlight
                splitrule = partitioner.get_partition_specs(ds_)
                
                for ii, rule in enumerate(splitrule):
                
                    partitioner = CustomPartitioner(splitrule=[rule],
                                                    attr=default_conf['split_attr']                                                
                                                    )
                
                    cvte = CrossValidation(clf,
                                           partitioner,
                                           splitter=Splitter(attr='partitions', attr_values=(2,3)),
                                           enable_ca=['stats', 'probabilities'])
                
                    sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
                    
                    
                    train_group = ds_.sa.group_split[ds_.sa[default_conf['split_attr']].value == rule[1][0]][0]
                    test_group = ds_.sa.group_split[ds_.sa[default_conf['split_attr']].value == rule[-1][0]][0]
                    test_subj = '_'.join(rule[-1])
                    
                    stringa = "Training Group: %s | Testing subject: %s | Testing Group: %s"
                    stringa = stringa % (train_group, test_subj, test_group)
                    logger.info(stringa)
                
                
                    sl_map = sl(ds_)
                    sl_map.samples *= -1
                    sl_map.samples +=  1

                    map_ = map2nifti(sl_map, imghdr=ds.a.imghdr)

                    maps.append(map_)
                    
                    
                    sl_fname = "sl_%s_%s_evidence_%s_split_%s_train_%s_test_%s_group_%s.nii.gz"  
                    sl_fname = sl_fname %  ('group', 
                                            task_, 
                                            str(ev), 
                                            str(ii+1), 
                                            train_group,
                                            test_subj,
                                            test_group)
                    
                    ni.save(map_, os.path.join(path, '0_results', 'group_sl', sl_fname))
                
                name = "%s_%s_%s_evidence_%s_balance_ds_%s" %('group', task_, data_type, str(ev), str(i+1))
                result_dict['radius'] = 3
                result_dict['map'] = map_
                
                subj_result = rs.SubjectResult(name, result_dict, savers)
                collection.add(subj_result)
            
            res.append(maps)
            
    return res