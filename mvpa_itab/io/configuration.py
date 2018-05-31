import os
import logging

logger = logging.getLogger(__name__)


def read_configuration (filename, section):
    """
    This function read the data configuration file
    the configuration file must be composed in this way
    
    
    .. configuration.conf

    [path]
    data_path=/media/robbis/DATA/fmri/carlo_ofp/
    subjects=/media/robbis/DATA/fmri/carlo_ofp/subjects.csv
    experiment=ofp
    types=OFP,RESIDUALS,OFP_NORES
    TR=1.7
    
    [OFP]
    sub_dir=analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise
    event_file=eventfile_beta_plus
    event_header=True
    img_pattern=residuals_sorted.nii.gz
    runs=1
    mask_dir=/media/robbis/DATA/fmri/carlo_ofp/1_single_ROIs
    brain_mask=glm_atlas_mask_333.nii.gz
        
    
    [roi_labels]
    lateral_ips=/media/robbis/DATA/fmri/carlo_ofp/1_single_ROIs
    """
    
    
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(filename)
    
    
    logger.info('Reading config file %s' %(filename))
    
    types = config.get('path', 'types').split(',')
    
    if types.count(section) > 0:
        types.remove(section)
    
    for typ in types:
        config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        if sec == 'roi_labels':
            roi_labels=dict()
            for k, v in config.items(sec):
                roi_labels[k] = v
        
            configuration.append(('roi_labels', roi_labels))
        
        for item in config.items(sec):
            configuration.append(item)
            logger.debug(item)
    
    
    cfg = dict(configuration)
    
    
    check_configuration(cfg)
    
    
    return cfg   



def check_configuration(cfg):
    
    mandatory_keys = ["subjects", "data_path", "event_file", "img_pattern", "brain_mask"]
    
    for key in mandatory_keys:
        if not key in cfg.keys():
            logger.error("No %s field in configuration" % (key))
            
            
        




def conf_to_json(config_file):
    
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    json_ = dict()
    
    for sec in config.sections():
        json_[sec] = dict()
        for item in config.items(sec):
            json_[sec][item[0]] = item[1]
    
    
    import json
    json_fname = file(config_file[:config_file.find('.')]+'.json', 'w')
    json.dump(json_, json_fname, indent=0)
    
    return json_



def read_json_configuration(path, json_fname, experiment):
    
    import json
    json_file = os.path.join(path, json_fname)
    
    conf = json.load(file(json_file, 'r'))
    
    experiments = conf['path']['types'].split(',')
    _ = [conf.pop(exp) for exp in experiments if exp != experiment]  
    
    print conf
    
    return conf


def read_remote_configuration(path):
        
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path, 'remote.conf'))
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            logger.debug(item)
    
    return dict(configuration) 

    logger.info('Reading remote config file '+os.path.join(path,'remote.conf'))