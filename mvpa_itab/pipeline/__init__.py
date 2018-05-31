import os
import nibabel as ni
import numpy as np
import mvpa_itab.results as rs
from mvpa_itab.script.carlo.analysis import carlo_memory_set_targets
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import debug
from mvpa_itab.io.base import read_configuration, load_subject_file
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.datasets.mri import map2nifti
from mvpa_itab.main_wu import detrend_dataset, normalize_dataset, slice_dataset,\
    change_target
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner

import logging
logger = logging.getLogger(__name__)




class SearchlightPipelineNL(object):
    """Searchligth Analysis class

    Provides basic interface for searchlight analysis, on the top of that
    other classes could be used to customize the pipeline with several
    searchlight flavours.

    Sequence of calls are:
    - pipeline.transform() -> pipeline.algorithm() -> pipeline.analysis() -> pipeline.searchlight()

    Parameters
    ----------
    data_loader : function that implements loading of PyMVPA file
        data_loader(configuration_file)
        
    preprocessing : preprocessing pipeline see ..preprocessing.pipelines
        
    analysis
    
    configuration_file : 
        

    Examples
    --------
    
    """
    def __init__(self, data_loader, analysis, configuration_file, **kwargs):
        
        self._data_loader_fx = data_loader
        
        object.__init__(self, *args, **kwargs)
        
        
    def fit(self, X, y):
        
        return
        
        





class SearchlightPipelinePyMVPA(object):
    """Searchligth Analysis class

    Provides basic interface for searchlight analysis, on the top of that
    other classes could be used to customize the pipeline with several
    searchlight flavours.

    Sequence of calls are:
    - pipeline.transform() -> pipeline.algorithm() -> pipeline.analysis() -> pipeline.searchlight()

    Parameters
    ----------
    kwargs : dictionary of analysis field
        

    Examples
    --------
    
    """
    def __init__(self, name="searchlight", storer=None):
        self.storer = storer
        self.name = name
        return
    
    
    def fit(self, ds, estimator, scorer, partitioner):
        
        # for each cross validation fold
        
        # fit the searchlight on training
        # score fitted searchlight on testing data
        # store results using storer
          
        
        
        return
    
    



class SearchlightAnalysisPipeline(object):
    """Searchligth Analysis class

    Provides basic interface for searchlight analysis, on the top of that
    other classes could be used to customize the pipeline with several
    searchlight flavours.

    Sequence of calls are:
    - pipeline.transform() -> pipeline.algorithm() -> pipeline.analysis() -> pipeline.searchlight()

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
                            'path':'/home/robbis/fmri/memory/', # Data
                            'configuration_file':"memory.conf", # Data
                            "project":"carlo_memory", # Data
                            "partecipants": "subjects.csv", # Data+Analysis
                            'data_type': 'BETA_MVPA', # Data
                            'n_folds':3, # Analysis
                            "condition_names":["evidence", "task"], # Data+Analysis
                            'evidence': 3, # Default_value (memory : 3) # Data+Analysis
                            'task':'decision', # Data
                            'split_attr':'subject', # Analysis
                            'mask_area':'intersect', # Data                          
                            'normalization':'both', # Analysis
                            "radius": 3, # Analysis
                            "n_balanced_ds": 1, # Analysis
                            "set_targets": carlo_memory_set_targets, # Data+Analysis (Outdated)
                            "classifier":LinearCSVMC(C=1) # Analysis (Not proper)
                            }
        
        
        
        
        self._default_conf.update(**kwargs)
        
        for k, v in self._default_conf.iteritems():
            setattr(self, "_"+k, v)
        
        
        # Setting default fields #
        if __debug__:
            debug.active += ["SLC"]
        
        
        self._conf = read_configuration(self._path, 
                                       self._configuration_file, 
                                       self._data_type)
        
        self._conf['analysis_type'] = 'searchlight'
        self._conf['analysis_task'] = self._project
        
        # Avoid non serializable objects saving
        self._default_conf.pop("set_targets")
        self._default_conf.pop("classifier")
        
        self._conf.update(**self._default_conf)
            
        self._data_path = self._conf['data_path']
        
        self.result_dict = dict()
        self.maps = []
        
        self._subjects, self._extra_sa = load_subject_file(os.path.join(self._path, self._partecipants))
    
    
    def searchlight(self, ds, cvte):
        
        sl = sphere_searchlight(cvte, 
                                radius= self._radius, 
                                space = 'voxel_indices')            
        sl_map = sl(ds)
        sl_map.samples *= -1
        sl_map.samples +=  1
        
        map_ = map2nifti(sl_map, imghdr=ds.a.imghdr)
        map_ = ni.Nifti1Image(map_.get_data(), affine=ds.a.imgaffine)

        self.maps.append(map_)
        
        return map_
    
    
    
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
        map_ = self.searchlight(ds, cvte)
         

        self.result_dict['radius'] = self._radius
        self.result_dict['map'] = map_
         
        subj_result = rs.SubjectResult(fname, 
                                       self.result_dict, 
                                       self._savers)
        
        self._collection.add(subj_result)
    
    
    
    def run(self, **options):
        """
        options: dictionary with condition to filter
            e.g. task="memory", evidence="1" depending on your data.
        """
        self._summarizers = [rs.SearchlightSummarizer()]
        self._savers = [rs.SearchlightSaver()]
        self._collection = rs.ResultsCollection(self._conf, 
                                            self._path, 
                                            self._summarizers,
                                            )
        
        summary_key = 'subjects'
        if 'summary_key' in options.keys():
            summary_key = options['summary_key']
        
        # Preprocess dataset with options
        ds = self.pre_operations(**options)

        # Get balancer (if count has changed it changes!)
        balance_generator = self.get_balancer(ds)
                               
        
        for i, ds_ in enumerate(balance_generator):
            self._conf["summary_ds_"+str(i)] = ds_.summary(chunks_attr=summary_key)
            logger.info(ds_.summary(chunks_attr=summary_key))
            self.algorithm(ds_, i)
            
        self._collection.summarize()
    
        
    
    
    def pre_operations(self, **options):
        
        # dictionary of conditions used
        self.conditions = {k: options[k] for k in self._condition_names}
        logger.debug(self._default_conf.keys())      

        # On-fly change default options
        # A little bit dangerous!!
        for k, v in options.iteritems():
            logger.debug(k)
            if k in self._default_conf.keys():
                setattr(self, "_"+k, v)
            else:
                setattr(self, k, v)


        ds = self.ds_orig.copy()
        
        ds = change_target(ds, options['task'])
        
        ds = detrend_dataset(ds, 
                            self._data_type, 
                            **self._conf)
        

        ds = slice_dataset(ds, options['condition'])
        
        ds = normalize_dataset(ds, **self._conf)

        return ds

