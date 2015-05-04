import os
import nibabel as ni
import numpy as np
import time
import logging


class Results(object):
    
    def __init__(self):
        self._result_array = []
        self.index = 0
        self.count = 0
    
    @classmethod
    def save(self, path, res_dir='0_results'):
        self.path = os.path.join(path, res_dir)
        make_dir(self.path)

    
    def add(self, result):
        if isinstance(result, SubjectResult):
            self._result_array.append(result)
            self.count+=1
    
    def next(self):
        if self.index < self.count:
            self.index+=1
            return self._result_array[self.index-1]
        else:
            raise StopIteration()
    
    def rewind(self):
        if self.count > 0:
            self.index = 0
        else:
            raise StopIteration()
          
        
class SearchlightResults(Results):
    
    expected_fields = ['name', 'map', 'radius']
    
    def __init__(self):
        Results.__init__()
        self._analysis = 'searchlight'
        self._total_map = []

    
    def save(self, path):
        
        if self.count == 0:
            raise ValueError()
        
        Results.save(self, path)
        
        for subj in self._result_array:
            
            if not self.is_correct(subj):
                raise ValueError()
            
            current_path = os.path.join(self.path, subj._name)
            make_dir(current_path)
            
            radius_ = np.int(subj._radius)
            map_ = subj._map
            name_ = subj._name
            
            if len(map.get_data().shape) > 3:
                mean_map = map_.get_data().mean(axis=3)
                mean_img = ni.Nifti1Image(mean_map, affine=map_.get_affine())
                fname = name_+'_radius_'+str(radius_)+'_searchlight_mean_map.nii.gz'
                ni.save(mean_img, os.path.join(current_path,fname))
            else:
                mean_map = map_.get_data()
                
            fname = name_+'_radius_'+str(radius_)+'_searchlight_map.nii.gz'
            ni.save(map_, os.path.join(current_path,fname))
            
            self._total_map.append(mean_map)
        
            self._total_map = np.array(self._total_map)
            total_img = ni.Nifti1Image(total_map, affine=map_.get_affine())
            fname = 'accuracy_map_radius_'+str(radius_)+'_searchlight_all_subj.nii.gz'
            ni.save(total_img, os.path.join(path,fname))
                       
            logging.info('Results writed in '+path)
            return path
        
    def is_correct(self, subj_results):
        subj_fields = set(subj_results.__dict__.keys())
        diff_header = set(subj_fields) - set(SearchlightResults.expected_fields)
            
        if len(diff_header) == 0:
            return True
        else:
            return False
        
    def summarize(self):
                
        if self.count == 0:
            raise ValueError()
        
        for subj in self._result_array:
            
            if not self.is_correct(subj):
                raise ValueError()
                        
            radius_ = np.int(subj._radius)
            map_ = subj._map
            name_ = subj._name
            
            if len(map.get_data().shape) > 3:
                mean_map = map_.get_data().mean(axis=3)
                mean_img = ni.Nifti1Image(mean_map, affine=map_.get_affine())
            else:
                mean_map = map_.get_data()
                            
            self._total_map.append(mean_map)
        
            self._total_map = np.array(self._total_map)
            total_img = ni.Nifti1Image(total_map, affine=map_.get_affine())
            fname = 'accuracy_map_radius_'+str(radius_)+'_searchlight_all_subj.nii.gz'
            ni.save(total_img, os.path.join(path,fname))
                       
            logging.info('Results writed in '+path)
            return path
            
            
            
            
            
        return None
        

class SubjectResult(object):
    
    def __init__(self, name, result_dict):
        self.name = name
        for k, v in result_dict.items():
            setattr(self, '_'+str(k), v)


def make_dir(self, path):
    """ Make dir unix command wrapper """
    command = 'mkdir '+os.path.join(path)
    os.system(command)
        
def save_image(self, filename, image):
    return  

def get_time():
    """Get the current time and returns a string (fmt: yymmdd_hhmmss)"""
    
    # Time acquisition
    tempo = time.localtime()
    
    datetime = ''
    i = 0
    for elem in tempo[:-3]:
        i = i + 1
        if len(str(elem)) < 2:
            elem = '0'+str(elem)
        if i == 4:
            datetime += '_'
        datetime += str(elem)
        
    return datetime 