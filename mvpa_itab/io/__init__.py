from mvpa_itab.preprocessing.pipelines import StandardPreprocessingPipeline, Node
from mvpa_itab.io.base import load_subject_ds
from mvpa_itab.io.configuration import read_configuration


class DataLoader(object):
    
    
    def __init__(self, configuration_file, task, reader=load_subject_ds, **kwargs):
        
        
        """
        This sets up the loading, a configuration file and a task is needed
        the task should be a section of the configuration file.        
        """        
        
        self._loader = reader
        self._configuration_file = configuration_file
        self._task = task
      
        
        self._conf = read_configuration(configuration_file, task)
        self._conf.update(**kwargs)
            
        self._data_path = self._conf['data_path']
        
        object.__init__(self, **kwargs)


        
        
    def fetch(self, prepro=StandardPreprocessingPipeline()):        
        
        return self._loader(self._configuration_file, self._task, prepro=prepro)
        