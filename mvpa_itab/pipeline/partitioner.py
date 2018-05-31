from mvpa2.suite import Partitioner
import numpy as np
import itertools
from mvpa2.generators.partition import CustomPartitioner


class TargetCombinationPartitioner(Partitioner):
    
    def __init__(self, attr_task='task', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__attr_task = attr_task
        
    
    def get_partition_specs(self, ds):
        #uniqueattr = ds.sa[self.__attr].unique()
        uniquetask = ds.sa[self.__attr_task].unique
        
        listattr = []
        for task in uniquetask:
            targets = ds[ds.sa[self.__attr_task].value == task].uniquetargets
            listattr.append(targets)                  
        print listattr
        rule = self._get_partition_specs(listattr)
        
        return rule
        
    def _get_partition_specs(self, listattr):
        
        prod = itertools.product(*listattr)
        
        return [('None', [item[0]], [item[1]])  for item in prod]
    
    

class SubjectGroupPartitioner(Partitioner):
    
    def __init__(self, attr_task='task', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__attr_task = attr_task
        
    
    def get_partition_specs(self, ds):
        #uniqueattr = ds.sa[self.__attr].unique()
        uniquetask = ds.sa[self.__attr_task].unique
        
        listattr = []
        for task in uniquetask:
            targets = ds[ds.sa[self.__attr_task].value == task].uniquetargets
            listattr.append(targets)                  
        print listattr
        rule = self._get_partition_specs(listattr)
        
        return rule
        
    def _get_partition_specs(self, listattr):
        
        prod = itertools.product(*listattr)
        
        return [('None', [item[0]], [item[1]])  for item in prod]
    
    
    
    
class MemoryGroupSubjectPartitioner(Partitioner):
    
    def __init__(self, group_attr='group_split', subject_attr='subject_split', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__group = group_attr
        self.__subject = subject_attr
        
    
    def get_partition_specs(self, ds):
        #uniqueattr = ds.sa[self.__attr].unique()
        unique_group = ds.sa[self.__group].unique
        unique_subj = ds.sa[self.__subject].unique
        
        listattr = []
        for group in unique_group:
            mask = ds.sa[self.__group].value == group
            group_sbj = np.unique(ds.sa[self.__subject].value[mask])
            left_subj = [s for s in unique_subj if not s in group_sbj]
            listattr.append([group_sbj, left_subj])                  
        rule = self._get_partition_specs(listattr)
        
        return rule
        
    def _get_partition_specs(self, listattr):
        
        splitrule = []
        for item in listattr:
            group = item[0]
            for s in item[1]:
                addrule = (['None'], group.tolist(), [s])
                splitrule.append(addrule)
                
        return splitrule


class LeaveOneSubjectPerGroupPartitioner(Partitioner):

    def __init__(self, group_attr='group_split', subject_attr='subject_split', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__group = group_attr
        self.__subject = subject_attr
        
    
    def get_partition_specs(self, ds):

        # Get unique values per group and subject
        unique_group = ds.sa[self.__group].unique
        unique_subj = ds.sa[self.__subject].unique
        
        listattr = []
        
        for subj in unique_subj:
            
            for group in unique_group:
                mask = ds.sa[self.__group].value == group
                
                # Get the subjects in the group
                group_sbj = np.unique(ds.sa[self.__subject].value[mask])
                group_sbj = [s for s in group_sbj if s != subj]              
                
                listattr.append([group_sbj, [subj]]) 
        
              
        rule = self._get_partition_specs(listattr)
        print rule
        return rule
        
    def _get_partition_specs(self, listattr):
        
        splitrule = []
        for item in listattr:
            group = item[0]
            for s in item[1]:
                addrule = (['None'], group, [s])
                splitrule.append(addrule)
                
        return splitrule
              
        
        
class GroupWisePartitioner(Partitioner):
    
    def __init__(self, attr_task='task', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__attr_task = attr_task
        
    
    def get_partition_specs(self, ds):
        #uniqueattr = ds.sa[self.__attr].unique()
        uniquetask = ds.sa[self.__attr_task].unique
        
        listattr = []
        for task in uniquetask:
            targets = ds[ds.sa[self.__attr_task].value == task].uniquetargets
            listattr.append(targets)                  
        print listattr
        rule = self._get_partition_specs(listattr)
        
        return rule
        
    def _get_partition_specs(self, listattr):
        
        prod = itertools.product(*listattr)
        
        return [
                (['3','4'],['1'],['2']),
                (['3','4'],['2'],['1']),
                (['1'],['2'],['3','4']),
                (['2'],['1'],['3','4']),
                
                (['1','2'],['3'],['4']),
                (['1','2'],['4'],['3']),
                (['3'],['4'],['1','2']),
                (['4'],['3'],['1','2']),
                ]        


class SKLearnCVPartitioner(CustomPartitioner):
    
    def __init__(self, sklearn_cv_obj, **kwargs):
        
        splitrule = []
        for _, test in sklearn_cv_obj:
            splitrule+=[(None, test)]
            
        CustomPartitioner.__init__(self, splitrule, **kwargs)
        
        
