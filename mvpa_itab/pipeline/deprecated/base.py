from mvpa_itab.pipeline.deprecated.searchlight import LeaveOneSubjectOutSL
import os
import mvpa_itab.results as rs
import numpy as np
import logging
from mvpa_itab.io.base import load_subjectwise_ds, read_configuration,\
    load_subject_file
from mvpa2.generators.resampling import Balancer
from mvpa2.generators.partition import CustomPartitioner, NFoldPartitioner
from mvpa2.measures.base import CrossValidation
from mvpa_itab.script.carlo.subject_wise_decoding import get_partitioner
from mvpa_itab.main_wu import detrend_dataset, normalize_dataset, spatial,\
    change_target, slice_dataset
from mvpa2.clfs.svm import LinearCSVMC
from mvpa_itab.script.carlo.analysis import carlo_memory_set_targets
from mvpa2.base import debug
logger = logging.getLogger(__name__)



class DecodingPipeline(object):
    """MVPA Pipeline

    Provides basic interface for roi-based analysis, on the top of that
    other classes could be used to customize the pipeline with several
    searchlight flavours.

    Sequence of calls are:
    - pipeline.run() 
    -> pipeline.pre_operations()
    -> pipeline.algorithm() 
    -> pipeline.analysis() 
    -> pipeline.decoding()

    Parameters
    ----------
    kwargs : dictionary of analysis field
        

    Examples
    --------
    
    """
        
  
    def __init__(self, name="searchlight", **kwargs):
        """
        This method is used to set up the configuration of the 
        analysis. 
        """
        
        self.name = name
        
        self._default_conf = {   
                            'path':'/home/robbis/fmri/memory/',
                            'configuration_file':"memory.conf",
                            "project":"carlo_memory",
                            "partecipants": "subjects.csv",
                            'data_type': 'BETA_MVPA',
                            'n_folds':3,
                            "condition_names":["evidence", "task"],
                            'evidence': 3, # Default_value (memory : 3)
                            'task':'decision', # Default_value
                            'split_attr':'subject', #
                            'mask_area':'intersect', # memory                            
                            'normalization':'both',
                            "radius": 3,
                            "n_balanced_ds": 1,
                            "set_targets": carlo_memory_set_targets,
                            "classifier":LinearCSVMC(C=1)
                            }
        
        
        
        
        self._default_conf.update(**kwargs)
        
        for k, v in self._default_conf.iteritems():
            setattr(self, "_"+k, v)
        
      
        
        self._conf = read_configuration(os.path.join(self._path, self._configuration_file), 
                                       self._data_type)
        
        self._conf['analysis_type'] = 'roi_based'
        self._conf['analysis_task'] = self._project
        
        # Avoid non serializable objects saving
        self._default_conf.pop("set_targets")
        self._default_conf.pop("classifier")
        
        self._conf.update(**self._default_conf)
            
        self._data_path = self._conf['data_path']
        
            
        self._subjects, _ = load_subject_file(os.path.join(self._path, self._partecipants))
        
        
        self.result_dict = dict()
        self.maps = []
        

    
    
    def _add_conf(self, name, obj):
        self._conf[name] = obj
    
    
    
    def decoding(self, ds, cvte):
        
        results = {}
        
        fa_keys = ds.fa.keys()
        
        
        if len(fa_keys) == 1:
            res = spatial(ds, cvte=cvte)
            results['roi'] = res
            return results
        
        
        fa_keys.remove('voxel_indices')
        for roi in fa_keys:
            
            for label in np.unique(ds.fa[roi])[1:]:
                
                logger.info("Analysis on roi %s %s" % (str(roi), str(label)))
                ds_ = ds[:, ds.fa[roi] == label]

                res = spatial(ds_, cvte=cvte)
                
                results["%s_%0.2d" %(roi, label)] = res
                   
        
        return results
    
    
    
    def build_fname(self, i):
        fname = self.name
        
        for k, v in self.conditions:
            fname += "_%s_%s" % (k, v)
            
        fname += "_balance_ds_%s.nii.gz" % (str(i+1))
        
        return fname 
        
        
    
    
    def analysis(self, ds, balance_ds_num, cvte=None, fname=None):
        
        if cvte == None:
            cvte = CrossValidation(self._classifier,
                                   NFoldPartitioner(cvtype=1),
                                   enable_ca=['stats', 'probabilities'])
        
        if fname == None:
            fname = self.build_fname(balance_ds_num)
        
        
        # Run the beast !! (cit.)
        res = self.decoding(ds, cvte)
         

        # Update results
        self.result_dict.update(res)
        
        for roi_name, result in res.iteritems():
            fname_subj = roi_name+'_'+fname
            subj_result = rs.SubjectResult(fname_subj, 
                                           result, 
                                           self._savers)
            
            self._collection.add(subj_result)
    
    
    
    def run(self, **options):
        """
        options: dictionary with condition to filter
            e.g. task="memory", evidence="1" depending on your data.
        """
        
        # Setup results
        self._summarizers = [rs.DecodingSummarizer()]
        self._savers = [rs.DecodingSaver()]
        self._collection = rs.ResultsCollection(self._conf, 
                                            self._path, 
                                            self._summarizers,
                                            **options)
        
        
        
        for subj in self._subjects:
        
            # Preprocess dataset with options
            ds = self.pre_operations(**options)
    
            # Get balancer (if count has changed it changes!)
            balance_generator = self.get_balancer(ds)
                                   
            
            for i, ds_ in enumerate(balance_generator):
                
                self._add_conf("summary_ds_%s" % (str(i)), 
                               ds_.summary(chunks_attr="subject"))

                
                logger.info(ds_.summary(chunks_attr="subject"))
                #logger.info(ds_.summary(target_attr='accuracy'))
                
                self.algorithm(ds_, i, subject=subj)
        
        
        if 'roi_labels' in self._conf.keys():
            for k, v in self._conf['roi_labels'].iteritems():
                if not isinstance(self._conf['roi_labels'][k], str):
                    self._conf['roi_labels'][k] = v.get_filename()
                
                   
        self._collection.summarize()
    
        
    
    def _modify_configuration(self, **options):
        # On-fly change default options
        # A little bit dangerous!!
        for k, v in options.iteritems():
            logger.debug(k)
            if k in self._default_conf.keys():
                setattr(self, "_"+k, v)
            else:
                setattr(self, k, v)
        
    
    
    def preprocessing(self, ds):
        
        ds = detrend_dataset(ds, 
                                self._data_type, 
                                **self._conf)
        
        ds = normalize_dataset(ds, **self._conf)
        
        return ds
    
    
    
    
    def pre_operations(self, **options):
        
        """
        This function is used to set
        """
        
        # dictionary of conditions used
        self.conditions = {k:options[k] for k in self._condition_names}
        logger.debug(self._default_conf.keys())      

        
        self._modify_configuration(**options)

        if not hasattr(self, 'ds_orig'):            
            self.ds_orig = self.load_dataset()
            
        ds = self.ds_orig.copy()
        
        # set the labels and returns new configuration
        
        ds = change_target(ds, options['task'])
        
        ds = detrend_dataset(ds, 
                            self._data_type, 
                            **self._conf)
        

        ds = slice_dataset(ds, options['condition'])
        
        ds = normalize_dataset(ds, **self._conf)

        return ds




class SingleSubjectDecoding(DecodingPipeline):
    """
    
    This class is used to perform single subject decoding
    this is the script used for start the analysis.
    
    from mvpa_itab.pipeline.base import LeaveOneSubjectOutDecoding
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
                'mask_area':'glm_atlas', # memory                            
                'normalization':'both',
                "radius": 3,
                "n_balanced_ds": 1,
                "set_targets": carlo_ofp_set_targets,
                "classifier":LinearCSVMC(C=1)
                }
                
    decoding_analysis = LeaveOneSubjectOutDecoding(**conf_ofp)
    decoding_analysis.load_dataset(**kwargs).transform(**options)
    """
    
    def __init__(self, name="decoding", **kwargs):
        return DecodingPipeline.__init__(self, name=name, **kwargs)
            
    
    
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
    
    
    
    
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit='subject') # mdm = 'group'
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    
    
    def algorithm(self, ds, balance_n, subject=None):
        
        ds = slice_dataset(ds, selection_dict={'subject':[subject]})
        
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
            
            fname = self.build_fname(ds, rule, ii, balance_n)

            self.analysis(ds, balance_n, cvte, fname)

            
        return
    
    





class LeaveOneSubjectOutDecoding(DecodingPipeline):
    """
    
    This class is used to perform leave one subject out decoding
    this is the script used for start the analysis.
    
    from mvpa_itab.pipeline.base import LeaveOneSubjectOutDecoding
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
                'mask_area':'glm_atlas', # memory                            
                'normalization':'both',
                "radius": 3,
                "n_balanced_ds": 1,
                "set_targets": carlo_ofp_set_targets,
                "classifier":LinearCSVMC(C=1)
                }
                
    decoding_analysis = LeaveOneSubjectOutDecoding(**conf_ofp)
    decoding_analysis.load_dataset(**kwargs).transform(**options)
    """
    
    def __init__(self, name="decoding_loso", **kwargs):
        return DecodingPipeline.__init__(self, name=name, **kwargs)
            
    
    
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
    
    
    
    
    
    def get_balancer(self, ds, method="pympva"):
        
        # TODO: Make also imbalanced-learn methods available
        balanc = Balancer(count=self._n_balanced_ds, 
                          apply_selection=True, 
                          limit='subject') # mdm = 'group'
        
        self.gen = balanc.generate(ds)
        
        return self.gen
    
    
    
    def algorithm(self, ds, balance_n):
        
        
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
            fname = self.build_fname(ds, rule, ii, balance_n)

            self.analysis(ds, balance_n, cvte, fname)

            
        return