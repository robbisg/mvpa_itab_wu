import os
import numpy as np

from mvpa2.generators.resampling import Balancer
from mvpa2.generators.partition import CustomPartitioner
from mvpa2.measures.base import CrossValidation
from mvpa2.misc.errorfx import mean_mismatch_error

from mvpa_itab.test_wu import subjects_merged_ds
from mvpa_itab.script.carlo.subject_wise_decoding import get_partitioner, SubjectWiseError
from mvpa_itab.wrapper.sklearn import SKLCrossValidation
from sklearn.cross_validation import StratifiedKFold
from mvpa_itab.pipeline import SearchlightAnalysisPipeline
from mvpa_itab.io.base import load_dataset

import logging
logger = logging.getLogger(__name__)



class LeaveOneSubjectOutSL(SearchlightAnalysisPipeline):
    
    def __init__(self, name="sl_loso", **kwargs):
        return SearchlightAnalysisPipeline.__init__(self, name=name, **kwargs)
            
       
    
    
    def load_dataset(self):
        
        
        ds_orig, _, _ = subjects_merged_ds(self._path,  # path or data_path
                                           None, 
                                           self._configuration_file, 
                                           self._data_type,
                                           subject_file=os.path.join(self._path, 
                                                                     self._partecipants),
                                           normalization=self._normalization,
                                           mask_area=self._mask_area)
        self.ds_orig = ds_orig
        return ds_orig
    
    
    
    def build_fname(self, ds, rule, split_no, balance_ds_no):
        """
        Example:
        
        sl_loso_cond1_val1_cond2_val2_split_1_test_2_train_1_balance_ds_1
        """
        
        train_group = ds.sa.group[ds.sa[self._split_attr].value == rule[1][0]][0]
        test_group  = ds.sa.group[ds.sa[self._split_attr].value == rule[-1][0]][0]
        test_subj = '_'.join(rule[-1])
        
        stringa = "Training Group: %s | Testing subject: %s | Testing Group: %s"
        stringa = stringa % (train_group, test_subj, test_group)
        logger.debug(stringa) # Log it!
        print stringa
        

        
        fname = self.name
        
        for k, v in self.conditions.iteritems():
            fname += "_%s_%s" % (k, v)
        
        
        fname += "_split_%s_train_%s_test_%s_group_%s_balance_ds_%s.nii.gz"  
        fname = fname %  (str(split_no+1),
                          train_group,
                          test_subj,
                          test_group,
                          str(balance_ds_no)
                          )
        
        return fname
    
    
    
    
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit='group')
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    
    def algorithm(self, ds, i):
        
        
        partitioner, splitter = get_partitioner(self._split_attr)
        splitrule = partitioner.get_partition_specs(ds)
        
        # With this (if the workstation runs out, we save results)      
        for ii, rule in enumerate(splitrule):
            
            print ii, rule
            ## Remove !?!!?
            
            print rule
            # We compose a simple partitioner
            partitioner = CustomPartitioner(splitrule=[rule],
                                            attr=self._split_attr                                                
                                            )
            # Custom cross-validator
            cvte = CrossValidation(self._classifier,
                                   partitioner,
                                   splitter=splitter,
                                   enable_ca=['stats', 'probabilities'],
                                   errorfx=SubjectWiseError(mean_mismatch_error, 
                                                            'group', 
                                                            'subject')
                                   )
            fname = self.build_fname(ds, rule, ii, i)
            print fname
            self.analysis(ds, i, cvte, fname)

            
        return
    
    
class SingleSubjectSearchlight(SearchlightAnalysisPipeline):
    
    def __init__(self, name="sl_single", **kwargs):
        SearchlightAnalysisPipeline.__init__(self, name=name, **kwargs)
        
    
    def load_dataset(self, subj):
        
        self.subject = subj
        
        ####
        
        for k in self._conf.keys():
            if k in self._default_conf.keys():
                self._conf[k] = self._default_conf[k]
        
        
        self.ds_orig = load_dataset(self._data_path, 
                                    self.subject, 
                                    self._data_type, 
                                    **self._conf)
        
        return self.ds_orig
        
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit=None)
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    
    def algorithm(self, ds, balance_ds_num, cvte=None, fname=None):
        
        # This is used for the sklearn crossvalidation
        y = np.zeros_like(ds.targets, dtype=np.int_)
        y[ds.targets == ds.uniquetargets[0]] = 1
        
        # We needs to modify the chunks in order to use sklearn
        ds.chunks = np.arange(len(ds.chunks))
        
        partitioner = SKLCrossValidation(StratifiedKFold(y, 
                                                         n_folds=self._n_folds))
        
        cvte = CrossValidation(self._classifier,
                               partitioner,
                               enable_ca=['stats', 
                                          'probabilities'])
        
        
        fname = self.build_fname(balance_ds_num)
        print fname
        self.analysis(ds, balance_ds_num, cvte, fname)
        
        
    
    
    def build_fname(self, i):
        
        fname = self.name+"_"+self.subject
        
        for k, v in self.conditions.iteritems():
            fname += "_%s_%s" % (k, v)
            
        fname += "_balance_ds_%s.nii.gz" % (str(i+1))
        
        return fname 
            

  
"""
sl_analysis = LeaveOneSubjectOutSL(path=")
sl_analysis.load_dataset(**kwargs).run(**options)
"""

            

            

                
                












