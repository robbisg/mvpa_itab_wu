import logging
import numpy as np
from mvpa2.generators.splitters import Splitter
from mvpa2.generators.partition import CustomPartitioner, NFoldPartitioner
from mvpa_itab.similarity.partitioner import MemoryGroupSubjectPartitioner
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.datasets.base import Dataset


#path = '/media/robbis/DATA/fmri/memory/'

logger = logging.getLogger(__name__)

def get_partitioner(split_attr='group_split'):
    
    splitter = Splitter(attr='partitions', attr_values=(2, 3))
    
    if split_attr == 'group_split':
    
        splitrule = [
                    # (leave, training, testing)
                    (['3','4'],['1'],['2']),
                    (['3','4'],['2'],['1']),
                    (['1'],['2'],['3','4']),
                    (['2'],['1'],['3','4']),
                    
                    (['1','2'],['3'],['4']),
                    (['1','2'],['4'],['3']),
                    (['3'],['4'],['1','2']),
                    (['4'],['3'],['1','2']),
                    ]
        partitioner = CustomPartitioner(splitrule=splitrule,
                                        attr=split_attr                                                
                                        )
                    
    elif split_attr == 'subject':
        
        partitioner = MemoryGroupSubjectPartitioner(group_attr='group_split', 
                                                    subject_attr=split_attr,
                                                    attr=split_attr)
        
    elif split_attr == "subject_ofp":
        
        partitioner = partitioner = NFoldPartitioner(attr="subject")
        splitter = Splitter(attr="partitions")
    
    elif split_attr == 'group':
        
        partitioner = NFoldPartitioner(attr=split_attr)
        splitter = Splitter(attr='partitions')
                
    return partitioner, splitter


def experiment_conf(task_, ev):
    
    conf = dict()
    # label managing
    if task_ == 'memory':
        conf['field'] = 'stim'
        conf['label_dropped'] = 'F0'
        conf['label_included'] = 'N'+ev+','+'O'+ev
        conf['count'] = 1
    else: # decision
        conf['field'] = 'decision'
        conf['label_dropped'] = 'FIX0'
        conf['label_included'] = 'NEW'+ev+','+'OLD'+ev
        conf['count'] = 1
        
    return conf


class SubjectWiseError(BinaryFxNode):
    
    def __init__(self, fx, space, subj_space, **kwargs):
        
        self.subj_space = subj_space
        BinaryFxNode.__init__(self, fx, space, **kwargs)
    
    def __call__(self, ds, _call_kwargs={}):     
        
        error_ = []
        subjects = np.unique(ds.sa[self.subj_space])
        for subj in subjects:
            subject_mask = ds.sa[self.subj_space].value == subj
            predictions = ds.samples[subject_mask].squeeze()
            targets = ds.sa.targets[subject_mask]
            err = self.fx(predictions, targets)
            error_.append(err)
             
        
        return Dataset(np.array(error_))
        



