import os
import nibabel as ni
import numpy as np
import mvpa_itab.results as rs
from mvpa_itab.script.carlo.analysis import carlo_memory_set_targets
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import debug
from mvpa_itab.io.base import read_configuration
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.datasets.mri import map2nifti
from mvpa_itab.main_wu import preprocess_dataset, normalize_dataset
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner

import logging
logger = logging.getLogger(__name__)



class SearchlightAnalysisPipeline(object):
    """Searchligth Analysis class

    Provides basic interface for searchlight analysis, on the top of that
    other classes could be used to customize the pipeline with several
    searchlight flavours.


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
                                            **options)
        
        # Preprocess dataset with options
        ds = self.pre_operations(**options)

        # Get balancer (if count has changed it changes!)
        balance_generator = self.get_balancer(ds)
                               
        
        for i, ds_ in enumerate(balance_generator):
            self._conf["summary_ds_"+str(i)] = ds_.summary(chunks_attr="subject")
            logger.info(ds_.summary(chunks_attr="subject"))
            self.algorithm(ds_, i)
            
        self._collection.summarize()
    
        
    
    
    def pre_operations(self, **options):
        
        # dictionary of conditions used
        self.conditions = {k:options[k] for k in self._condition_names}
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
        
        targets, self._conf = self._set_targets(ds, 
                                                self._conf, 
                                                **self.conditions)

        ds.targets = targets
        
        ds = preprocess_dataset(ds, 
                                self._data_type, 
                                **self._conf)
        
        ds = normalize_dataset(ds, **self._conf)

        return ds

