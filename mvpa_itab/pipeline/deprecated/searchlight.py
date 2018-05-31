import os
import numpy as np

from mvpa2.generators.resampling import Balancer
from mvpa2.generators.partition import CustomPartitioner
from mvpa2.measures.base import CrossValidation

from mvpa_itab.io.base import load_subjectwise_ds
from mvpa_itab.script.carlo.subject_wise_decoding import get_partitioner

from sklearn.cross_validation import StratifiedKFold
from mvpa_itab.pipeline import SearchlightAnalysisPipeline
from mvpa_itab.io.base import load_dataset

import logging
from mvpa_itab.pipeline.partitioner import SKLearnCVPartitioner
logger = logging.getLogger(__name__)



class LeaveOneSubjectOutSL(SearchlightAnalysisPipeline):
    """
    
    This class is used to perform leave one subject out searchlight
    this is the script used for start the analysis.
    
    from mvpa_itab.pipeline.searchlight import LeaveOneSubjectOutSL
    from mvpa_itab.script.carlo.analysis import carlo_ofp_set_targets
    from mvpa2.suite import LinearCSVMC
    
    conf_ofp = {   
                'path':'/media/robbis/DATA/fmri/carlo_ofp/',
                'configuration_file':"ofp.conf",
                "project":"carlo_ofp",
                "partecipants": "subjects.csv",
                'data_type': 'OFP',
                'n_folds':3,
                "condition_names":["evidence", "task"],
                'evidence': 3, # Default_value (memory : 3)
                'task':'decision', # Default_value
                'split_attr':'subject_ofp', #
                'mask_area':'glm_atlas_mask_333.nii.gz', # memory                            
                'normalization':'both',
                "radius": 3,
                "n_balanced_ds": 1,
                "set_targets": carlo_ofp_set_targets,
                "classifier":LinearCSVMC(C=1)
                }
                
    sl_analysis = LeaveOneSubjectOutSL(**conf_ofp)
    sl_analysis.load_dataset(**kwargs).run(**options)
    """
    
    
    
    def __init__(self, name="sl_loso", **kwargs):
        return SearchlightAnalysisPipeline.__init__(self, name=name, **kwargs)
    
    
    def algorithm(self, ds, i):
        
        
        partitioner, splitter = get_partitioner(self._split_attr)
        splitrule = partitioner.get_partition_specs(ds)
        print splitrule
        # With this (if the workstation runs out, we save results)      
        for ii, rule in enumerate(splitrule):
            
            logger.debug(rule)
            
            # We compose a simple partitioner
            partitioner = CustomPartitioner(splitrule=[rule],
                                            attr="subject"                                                
                                            )
            # Custom cross-validator
            # Watch out of Subject wise error
            # TODO: Use a strategy for errorfx
            cvte = CrossValidation(self._classifier,
                                   partitioner,
                                   splitter=splitter,
                                   enable_ca=['stats', 'probabilities'],
                                   #errorfx=SubjectWiseError(mean_mismatch_error, 
                                   #                         'group', 
                                   #                         'subject')
                                   )
            fname = self.build_fname(ds, rule, ii, i)

            self.analysis(ds, i, cvte, fname)

            
        return
            
            
            
    def build_fname(self, ds, rule, split_no, balance_ds_no):
        """
        Example:
        
        sl_loso_cond1_val1_cond2_val2_split_1_test_2_train_1_balance_ds_1
        """
        print rule
        _split_attr = "subject" # OFP
        #_split_attr = "group"
        train_group = ds.sa.group[ds.sa[_split_attr].value == rule[1][0]][0]
        test_group  = ds.sa.group[ds.sa[_split_attr].value == rule[-1][0]][0]
        test_subj = '_'.join(rule[-1])
        
        stringa = "Training Group: %s | Testing subject: %s | Testing Group: %s"
        stringa = stringa % (train_group, test_subj, test_group)
        logger.debug(stringa) # Log it!
        logger.info(stringa)
        

        
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
    
    
    def check_conf(self, value):
        
        if value in self._conf:
            return self._conf[value]
        
        return None
    
    
    
    def load_dataset(self):
        
        
        label_name = self.check_conf("label_mask_name")
        label_value = self.check_conf("label_mask_value")
        roi_labels = self.check_conf("roi_labels")
                    
        print label_name, label_value
        
        ds_orig, _, _ = load_subjectwise_ds(self._path,  # path or data_path
                                           os.path.join(self._path, 
                                                        self._partecipants), 
                                           self._configuration_file, 
                                           self._data_type,
                                           normalization=self._normalization,
                                           mask_area=self._mask_area,
                                           label_mask_name=label_name,
                                           label_mask_value=label_value,
                                           roi_labels=roi_labels
                                           )
        self.ds_orig = ds_orig
        
        self._conf["summary_ds_load"] = ds_orig.summary(chunks_attr="subject")
        logger.info(ds_orig.summary(chunks_attr="subject"))
        
        return ds_orig
    
     
    
    
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit='subject') # mdm = 'group'
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    

    
    
class SingleSubjectSearchlight(SearchlightAnalysisPipeline):
    
    
    
    def __init__(self, name="sl_single", **kwargs):
        SearchlightAnalysisPipeline.__init__(self, name=name, **kwargs)



    def algorithm(self, ds, balance_ds_num, cvte=None, fname=None):
        
        # This is used for the sklearn crossvalidation
        y = np.zeros_like(ds.targets, dtype=np.int_)
        y[ds.targets == ds.uniquetargets[0]] = 1
        
        # We needs to modify the chunks in order to use sklearn
        ds.chunks = np.arange(len(ds.chunks))

        b = balance_ds_num
        self._conf["summary_ds_"+str(b)] = ds.summary()
        logger.info(ds.summary())
        
        partitioner = SKLearnCVPartitioner(
                            StratifiedKFold(y, 
                                            n_folds=self._n_folds)
                                        )
        
        cvte = CrossValidation(self._classifier,
                               partitioner,
                               enable_ca=['stats', 
                                          'probabilities'])
        
        
        fname = self.build_fname(balance_ds_num)

        self.analysis(ds, balance_ds_num, cvte, fname)
        
        
    
    
    def build_fname(self, i):
        
        fname = self.name+"_"+self.subject
        
        for k, v in self.conditions.iteritems():
            fname += "_%s_%s" % (k, v)
            
        fname += "_balance_ds_%s.nii.gz" % (str(i+1))
        
        return fname 
            
        
    
    def load_dataset(self, subj):
        
        self.subject = subj
        
        analysis_dict = {}
        for k in self._conf.keys():
            if k in self._default_conf.keys():
                self._conf[k] = self._default_conf[k]
            if k != 'path' and k != 'task':
                analysis_dict[k] = self._conf[k]
        
        
        self.ds_orig = load_dataset(self._data_path, 
                                    self.subject, 
                                    self._data_type, 
                                    **analysis_dict)
        
        return self.ds_orig
        
    
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit=None)
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    


  


