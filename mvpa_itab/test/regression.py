import unittest
from mvpa_itab.regression.base import *
import os
import nibabel as ni
import numpy as np
from sklearn.linear_model.coordinate_descent import Lasso, ElasticNet

class TestRegression(unittest.TestCase):
    
    
    def test_features(self):
        algorithms_ = [SVR(kernel='linear', C=1), 
                       #SVR(kernel='rbf', C=1),
                       #SVR(kernel='poly', C=1, degree=3),
                       SVR(kernel='linear', C=10),
                       SVR(kernel='linear', C=0.5),
                       #ElasticNet(alpha=0.5, fit_intercept=True),
                       #Lasso(alpha=0.5, fit_intercept=True)
                       ]
        conditions_ = ['Samatha', 'Vipassana']

        roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                              delimiter=',',
                              dtype=np.str)
        
        iterator_setup = {'directory':
                          ['20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri',
                           '20150427_124039_connectivity_fmri'],
                          'meditation': conditions_,
                          'learner': algorithms_
                          
                          }
        
        subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                              dtype=np.str)
        
        
        
        
        
        _fields = {'path':'/media/robbis/DATA/fmri/monks/', 
                       'roi_list':
                            np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                            delimiter=',',
                            dtype=np.str), 
                       'directory':'20150427_124039_connectivity_fmri', 
                       'condition_list':['Samatha', 'Vipassana'],
                       'subjects':subjects,
                       'filter_':{'meditation':'Samatha', 'groups':'E'},
                       'fs_algorithm':Correlation,
                       'fs_ranking_fx':ranking_correlation,
                       'cv_schema':ShuffleSplit(12, 
                                                n_iter=50, 
                                                test_size=0.25),
                       'learner':SVR(kernel='linear', C=1),
                       'error_fx':[mean_squared_error, correlation],
                       'y_field':'expertise',
                       'n_permutations':1000
                       }
        
        
        
        
        return
    

if __name__ == '__main__':
    unittest.main()