import unittest
from mvpa_itab.connectivity import ConnectivityPreprocessing
import os
import nibabel as ni
import numpy as np

class TestConnectivity(unittest.TestCase):
    def test_data(self):
        path = '/media/robbis/DATA/fmri/monks'
        subj = '061102chrwoo'
        connect = ConnectivityPreprocessing(path, 
                                            subj, 
                                            'bold_orient.nii.gz',
                                            'bold_orient_mask_mask.nii.gz',
                                            ['mprage_orient_brain_seg_wm_333.nii.gz',
                                             'mprage_orient_brain_seg_csf_333.nii.gz']
                                            )
        file_ = connect.execute()
        
        test_file = ni.load(os.path.join(path, subj, 'fmri', 'residual_filtered_gsr.nii.gz'))
        
        np.testing.assert_array_almost_equal(file_.shape, test_file.get_data().shape)
        
        np.testing.assert_array_almost_equal(file_, test_file.get_data())
    

if __name__ == '__main__':
    unittest.main()
    