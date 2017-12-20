import os
import logging

logger = logging.getLogger(__name__)


def read_configuration (path, filename, section):
    
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path, filename))
    
    
    logger.info('Reading config file '+os.path.join(path,filename))
    
    types = config.get('path', 'types').split(',')
    
    if types.count(section) > 0:
        types.remove(section)
    
    for typ in types:
        config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            logger.debug(item)
    
    return dict(configuration)   



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