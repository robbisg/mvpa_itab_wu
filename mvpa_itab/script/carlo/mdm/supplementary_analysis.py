_default_options =  [
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"]},              
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 40, "O": 40}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [6])],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 40, "O": 40}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 40, "S": 40}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                    ]



_default_config = {
                    'prepro': [
                               'target_transformer',
                               'sample_slicer', 
                               'balancer'
                               ],
                    "balancer__attr": 'subject',
                    
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))
                        ],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    #'cv__n_splits': 7,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': TemporalDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    #'kwargs__roi': ['omnibus', 'decision', 'image+type', 'motor+resp', 'target+side'],
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'

                    }



iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()

    a = AnalysisPipeline(conf, name="temporal_decoding_across_fsel").fit(ds, **kwargs)
    a.save()
    gc.collect()

##################################################
loader = DataLoader(configuration_file=configuration_file, 
                    #data_path="/home/carlos/mount/meg_workstation/Carlo_MDM/",
                    task='BETA_MVPA',
                    roi_labels=roi_labels,
                    event_file="beta_attributes_full", 
                    brain_mask="mask_intersection")

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      ])
#prepro = PreprocessingPipeline()

ds = loader.fetch(prepro=prepro)
ds = MemoryReducer(dtype=np.float16).transform(ds)



_default_options = {
    'target_transformer__attr': ['decision', 'memory_status']
}



_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    'sample_slicer__attr': {'decision':["N", "O"],'evidence':[1]},
                    "balancer__attr": 'subject',
                    
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                           ('decision', [4]), ('decision', [5]),
                                            ('decision', [6]), ('decision', [7]), ('decision', [8]),
                                           ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                           ('motor+resp', [4]), ('motor+resp', [5])('motor+resp', [6])],
                     }
                    'kwargs__cv_attr': 'subject'
                    }

import gc 
iterator = AnalysisIterator(_default_options,  
                            AnalysisConfigurator(**_default_config)) 
for conf in iterator: 
    kwargs = conf._get_kwargs() 
 
    a = AnalysisPipeline(conf, name="roi_decoding_across_memoryvsdecision1x").fit(ds, **kwargs) 
    a.save() 
    gc.collect()



###################################################
_default_options =  [
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [6])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),        
                      'kwargs__roi_values': [('image+type', [6])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[5]},                 
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [6])],
                     },
                     
                     ################################################################
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),       
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     },
                     #######################################################################
                     {         
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),              
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [6]), ('decision', [7]), ('decision', [8]),
                                             ('decision', [9]), ('decision', [10]),
                                             ('motor+resp', [6])],
                     }
                     }
                    ]




_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    "balancer__attr": 'subject',
                    
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    #'kwargs__roi': labels,
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'
                    }

import gc 
iterator = AnalysisIterator(_default_options,  
                            AnalysisConfigurator(**_default_config), 
                            kind='configuration') 
for conf in iterator: 
    kwargs = conf._get_kwargs() 
 
    a = AnalysisPipeline(conf, name="roi_decoding_across_full").fit(ds, **kwargs) 
    a.save() 
    gc.collect()