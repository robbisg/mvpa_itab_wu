import numpy as np
from scipy.io import loadmat

class MEGData :
    
    def __init__(self):
        
        return
    
    def __call__(self):
        
        return
    
class MEGHeader:
    
    def __init__(self):
        
        self.fmt = None

        
        return
    
    def __call__(self):
        
        return
    
    @classmethod
    def from_filename(klass, hdr_dict):
        
        
        var_names = hdr_dict.dtype.names
        
        for k in var_names:
            
            if k == 'signal_format':
                if hdr_dict[k] == u'float':
                    setattr(self, 'fmt', np.dtype('float32'))
            if k ==       
            
            
            setattr(self, key, initial_data[key])

def load_meg(data_fname, header_fname):
    # Do we need to open header here o we pass the parsed header
    
    hdr_dict = loadmat(header_fname)
    hdr = hdr_dict['header'][0][0]
    
    

    _, ext = data_fname.split('.')
    if ext != 'dat':
        raise ValueError('Only .dat files are supported!')

    with open(data_fname, 'rb') as fid:
        data_array = np.fromfile(fid, fmt)
        
        
        
        
        