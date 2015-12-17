import numpy as np
from scipy.stats.stats import zscore
from mvpa_itab.stats import Correlation
from mvpa_itab.measure import ranking_correlation
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_squared_error
from mvpa_itab.measure import correlation
from mvpa_itab.regression.base import ConnectivityDataLoader, FeatureSelectionIterator, \
                                        RegressionAnalysis, PermutationAnalysis


class Pipeline(object):
    
    def update_configuration(self, **kwargs):
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            
            if arg in self.filter_.keys():
                self.filter_[arg] = kwargs[arg]
                
        return self
    
    
    def save(self):
        import json
        import os
        
        js_dict = dict()
        js_dict['conf'] = str(self._configuration)
        js_dict['data'] = np.array(self.results).tolist()
        
        fname = os.path.join(self.path, 
                            '0_results', 
                            self.directory,
                            'result.json')
        
        with open(fname, 'w') as fp:
            json.dump(js_dict, fp)
            
            
    def __str__(self, *args, **kwargs):
        self.__name__

class RegressionAnalysisPipeline(Pipeline):
    
    _default_fields = {'path':'/media/robbis/DATA/fmri/monks/', 
               'roi_list':None, 
               'directory':None, 
               'condition_list':None,
               'subjects':None,
               'filter_':None,
               'fs_algorithm':Correlation,
               'fs_ranking_fx':ranking_correlation,
               'cv_schema':ShuffleSplit(12, 
                                        n_iter=250, 
                                        test_size=0.25),
               'learner':SVR(kernel='linear', C=1),
               'error_fx':[mean_squared_error, correlation],
               'y_field':None,
               'n_permutations':2000
               }
    
    
    def setup_analysis(self, **kwargs):
        """
        Fields needed:
        
        path: (e.g. '/media/robbis/DATA/fmri/monks/')
        roi_list: (e.g. load '/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt')
        directory (e.g. '20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri')
        condition_list (e.g. ['Samatha', 'Vipassana'])
        subjects  (e.g. '/media/robbis/DATA/fmri/monks/attributes_struct.txt')
        filter  (e.g. {'meditation':'Samatha', 'groups':'E'})
            -- filter_keywords were used to build the filter
                e.g. meditation: Vipassana will overwrite the filter
        fs_algorithm  (e.g. mvpa_itab.stats.Correlation)
        fs_ranking_fx (e.g. mvpa_itab.measures.ranking_correlation)
        cv_schema (e.g. ShuffleSplit(num_exp_subjects, n_iter=cv_repetitions, test_size=cv_fraction)
        learner (e.g. SVR(kernel='linear', C=1))
        error_fx (e.g. [mean_squared_error, correlation])
        y_field (e.g. expertise)
        n_permutations (e.g. 1000)
        """
        self._configuration = dict()
        for arg in self._default_fields:
            setattr(self, arg, self._default_fields[arg])
            self._configuration[arg] = self._default_fields[arg]
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
        
        return self


    def update_configuration(self, **kwargs):
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
            if arg in self.filter_.keys():
                self.filter_[arg] = kwargs[arg]
                self._configuration['filter_'][arg] = kwargs[arg]
                
        return self
    

            
    def run(self):
        
        self.results = []
        self.loader = ConnectivityDataLoader()
        X, y = self.loader.setup_analysis(self.path, 
                              self.roi_list, 
                              self.directory, 
                              self.condition_list, 
                              self.subjects).filter(self.filter_).get_data()


        X = zscore(X, axis=1) # Sample-wise
        y = zscore(np.float_(y))
        
        fs = FeatureSelectionIterator()
        fs.setup_analysis(self.fs_algorithm, self.fs_ranking_fx).run(X, y).select_first(0.1)
        
        reg = RegressionAnalysis().setup_analysis(self.cv_schema, 
                                                  self.learner, 
                                                  self.error_fx)
        
        perm = PermutationAnalysis().setup_analysis(reg, 
                                                    n_permutation=self.n_permutations)
        

        
        for set_ in fs:
            X_ = X[:,set_]
            y_ = y
                        
            reg_res = reg.run(X_, y_) # To be selected
            n_dist = perm.run(X_, y_)
            
            p_res = perm.pvalues(reg_res)
            
            self.results.append([reg_res, n_dist, p_res])

        #self.save()

        

class NoPermutationPipeline(RegressionAnalysisPipeline):
    
    
    def run(self):
        
        self.loader = ConnectivityDataLoader()
        X, y = self.loader.setup_analysis(self.path, 
                              self.roi_list, 
                              self.directory, 
                              self.condition_list, 
                              self.subjects).filter(self.filter_).get_data()
                              
        X = zscore(X, axis=1) # Sample-wise
        y = zscore(np.float_(y))       
        
        fs = FeatureSelectionIterator()
        fs.setup_analysis(self.fs_algorithm, 
                          self.fs_ranking_fx).run(X, y).select_first(0.1)
        
        reg = RegressionAnalysis().setup_analysis(self.cv_schema, 
                                                  self.learner, 
                                                  self.error_fx)
        self.results = []
        for set_ in fs:
            X_ = X[:,set_]
            y_ = y
                        
            reg_res = reg.run(X_, y_) # To be selected
                       
            self.results.append([reg_res])
        
        self.save()
        
        return self.results
            

class MemoryPipeline(Pipeline):
    
    _default_fields = {'path':'/media/robbis/DATA/fmri/memory/', 
               'roi_list':None, 
               'directory':None, 
               'condition_list':None,
               'subjects':None,
               'task':None,
               'fs_algorithm':Correlation,
               'fs_ranking_fx':ranking_correlation,
               'cv_schema':ShuffleSplit(12, 
                                        n_iter=250, 
                                        test_size=0.25),
               'learner':SVR(kernel='linear', C=1),
               'error_fx':[mean_squared_error, correlation],
               'y_field':None,
               'n_permutations':2000
               }
    
    def setup_analysis(self, **kwargs):
        """
        Fields needed:
        
        path: (e.g. '/media/robbis/DATA/fmri/memory/')
        roi_list: (e.g. load '/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt')
        directory (e.g. '20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri')
        condition_list (e.g. ['Samatha', 'Vipassana'])
        subjects  (e.g. '/media/robbis/DATA/fmri/monks/attributes_struct.txt')
        filter  (e.g. {'meditation':'Samatha', 'groups':'E'})
            -- filter_keywords were used to build the filter
                e.g. meditation: Vipassana will overwrite the filter
        fs_algorithm  (e.g. mvpa_itab.stats.Correlation)
        fs_ranking_fx (e.g. mvpa_itab.measures.ranking_correlation)
        cv_schema (e.g. ShuffleSplit(num_exp_subjects, n_iter=cv_repetitions, test_size=cv_fraction)
        learner (e.g. SVR(kernel='linear', C=1))
        error_fx (e.g. [mean_squared_error, correlation])
        y_field (e.g. expertise)
        n_permutations (e.g. 1000)
        """
        
        self._configuration = dict()

        for arg in self._default_fields:
            setattr(self, arg, self._default_fields[arg])
            self._configuration[arg] = self._default_fields[arg]
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
        
        return self

    
    
    def update_configuration(self, **kwargs):
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
            if arg in self.filter_.keys():
                self.filter_[arg] = kwargs[arg]
                self._configuration['filter_'][arg] = kwargs[arg]
                
        return self
    
    
    def run(self):
        
        conf = read_configuration(path, 'remote_memory.conf', data_type)
        #conf['mask_area'] = 'PCC'
        ds = load_dataset(path, subj, data_type, **conf)
        
        # label managing
        # ds.targets = ds.sa.memory_status
        ds.targets = np.core.defchararray.add(np.array(ds.sa.decision, dtype=np.str), 
                                              #np.array(ds.sa.stim, dtype=np.str),
                                              np.array(ds.sa.evidence,dtype= np.str))
        ev = str(ev)
        
        conf['label_dropped'] = 'FIX0'
        conf['label_included'] = 'NEW'+ev+','+'OLD'+ev
        
        ds = preprocess_dataset(ds, data_type, **conf)

        balanc = Balancer(count=1, apply_selection=True)
        gen = balanc.generate(ds)
        #clf = AverageLinearCSVM(C=1)
        
        clf = LinearCSVMC(C=1)
        #avg = TrialAverager(clf)
        cv_storage = StoreResults()
        
        skclf = SVC(C=1, kernel='linear', class_weight='auto')
        clf = SKLLearnerAdapter(skclf)
        
        cvte = CrossValidation(clf, 
                               NFoldPartitioner(cvtype=2),
                               #errorfx=ErrorPerTrial(), 
                               #callback=cv_storage,
                               enable_ca=['stats', 'probabilities'])

        sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
        maps = []
        
        
        for i, ds_ in enumerate(gen):
            
            #Avoid balancing!
            ds_ = ds
            
            sl_map = sl(ds_)
            sl_map.samples *= -1
            sl_map.samples +=  1
            #sl_map.samples = sl_map.samples.mean(axis=len(sl_map.shape)-1)
            map_ = map2nifti(sl_map, imghdr=ds.a.imghdr)
            maps.append(map)
            
            name = "%s_%s_evidence_%s_balance_ds_%s" %(subj, data_type, str(ev), str(i+1))
            result_dict['radius'] = 3
            result_dict['map'] = map_
            
            subj_result = rs.SubjectResult(name, result_dict, savers)
            collection.add(subj_result)
        
        res.append(maps)
        
        
        return
    
    
    
    
    
    