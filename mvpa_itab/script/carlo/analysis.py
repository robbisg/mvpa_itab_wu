import numpy as np  
               

def carlo_memory_set_targets(ds, configuration, **conditions):
    
    
    evidence = conditions['evidence']
    task = conditions['task']
    
    
    conf = dict()
    # label managing
    if task == 'memory':
        conf['field'] = 'stim'
        conf['label_dropped'] = 'F0'
        conf['label_included'] = 'N'+evidence+','+'O'+evidence
    else: # decision
        conf['field'] = 'decision'
        conf['label_dropped'] = 'FIX0'
        conf['label_included'] = 'NEW'+evidence+','+'OLD'+evidence
    
    
    field_ = conf.pop('field')
    configuration.update(conf)
    
    
    targets = np.core.defchararray.add(np.array(ds.sa[field_].value, 
                                                   dtype=np.str), 
                                       np.array(ds.sa.evidence, 
                                                   dtype= np.str))
    
    return targets, configuration



