#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
import os
from mvpa2.suite import find_events, fmri_dataset, SampleAttributes
from mvpa2.suite import eventrelated_dataset
import logging
import time
import numpy as np
import nibabel as ni
from mvpa_itab.fsl_wrapper import bet_wu_data_
from mvpa_itab.utils import fidl_convert
#from memory_profiler import profile

logger = logging.getLogger(__name__)

def get_time():
    
    """Get the current time and returns a string (fmt: yymmdd_hhmmss)
    
    !!! THIS HAS BEEN INCLUDED IN results MODULE !!!
    
    """
    
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


#@profile
def load_dataset(path, subj, folder, **kwargs):
    ''' Load file given filename, the 

    Parameters
    ----------
    path : string
       specification of filepath to load
    subj : string
        subject name (in general it specifies a subfolder under path)
    folder : string
        subfolder under subject folder (in general is the experiment name)
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    ds : ``Dataset``
       Instance of ``mvpa2.datasets.Dataset``
    '''

    # TODO: Slim down algorithm parts and checks
    
    use_conc = 'False'
    skip_vols = 0
    ext=''
    
    logger.debug(str(kwargs))
    
    for arg in kwargs:
        if arg == 'skip_vols':              # no. of canceled volumes
            skip_vols = np.int(kwargs[arg])
        if arg == 'use_conc':               # .conc file used (True/False)
            use_conc = kwargs[arg]
        if arg == 'conc_file':              # .conc pathname
            conc_file = kwargs[arg]
        if arg == 'sub_dir':                # subdirs to look for files
            sub_dir = kwargs[arg].split(',')
        if arg == 'img_extension':          # image extension
            ext = kwargs[arg]
    
    # Load the filename list        
    if use_conc == 'False':
        file_list = load_wu_file_list(path, name=subj, task=folder, **kwargs)   
    else:
        file_list = read_conc(path, subj, conc_file, sub_dir=sub_dir)
        file_list = modify_conc_list(path, subj, file_list, extension=ext)

        kwargs = update_subdirs(file_list, subj, **kwargs) # updating args

    # Load data
    try:
        fmri_list = load_fmri(file_list, skip_vols=skip_vols)
    except IOError, err:
        logger.error(err)
        return 0
    
    ### Code to substitute   
    attr = load_attributes(path, folder, subj, **kwargs)              

    # Loading mask 
    mask = load_mask(path, subj, **kwargs)        
       
    # Check attributes/dataset sample mismatches
    vol_sum = np.sum([img.shape[3] for img in fmri_list])

    if vol_sum != len(attr.targets):
        logger.debug('Volumes no.: '+str(vol_sum)+' Targets no.: '+str(len(attr.targets)))
        del fmri_list
        logger.error(subj + ' *** ERROR: Attributes Length mismatches with fMRI volumes! ***')
        raise ValueError('Attributes Length mismatches with fMRI volumes!')       
    
    # Load the pymvpa dataset.
    try:
        logger.info('Loading dataset...')
        ds = fmri_dataset(fmri_list, targets=attr.targets, chunks=attr.chunks, mask=mask) 
        logger.info('Dataset loaded...')
    except ValueError, e:
        logger.error(subj + ' *** ERROR: '+ str(e))
        del fmri_list
        return 0;
    
    # Update Dataset attributes
    #
    # TODO: Evaluate if it is useful to build a dedicated function
    ev_list = []
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)
    for i in range(len(events)):
        duration = events[i]['duration']
        for j in range(duration):
            ev_list.append(i+1)
               
    ds.a['events'] = events  # Update event field
    ds.sa['events_number'] = ev_list # Update event number
    
    # Name added to do leave one subject out analysis
    ds.sa['name'] = [subj for i in range(len(ds.sa.chunks))] 
    
    try:
        for k in attr.keys():
            ds.sa[k] = attr[k]
    except BaseException, e:
        logger.error('attributes not found.')
         
    f_list = []
    for i, img_ in enumerate(fmri_list):
        f_list += [i+1 for _ in range(img_.shape[-1])]
        
    # For each volume indicates to which file it belongs
    # It is used for detrending!
    ds.sa['file'] = f_list
    
    del fmri_list
    
    return ds 

#@profile    
def load_wu_file_list(path, name, task, el_vols=None, **kwargs):
    ''' Load file given filename, the 

    Parameters
    ----------
    path : string
       specification of filepath to load
    name : string
        subject name (in general it specifies a subfolder under path)
    task : string
        subfolder under subject folder (in general is the experiment name)
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    file_list : string list
       list of strings indicating the file pathname
    '''
    
    # TODO: 
    
    # What does it means analysis=single???
    analysis = 'single'
    img_pattern=''
    
    for arg in kwargs:
        if (arg == 'img_pattern'):
            img_pattern = kwargs[arg] 
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        if (arg == 'runs'):
            runs = int(kwargs[arg])
        if (arg == 'use_task_name'):
            use_task_name = kwargs[arg]

    path_file_dirs = []
    
    if use_task_name == 'False':
        task = ''
    
    for dir_ in sub_dirs:
        if dir_ == 'none':
            dir_ = ''
        if dir_.find('/') == -1:
            logger.debug(dir_)
        path_file_dirs.append(os.path.join(path,name,dir_))

   
    logger.info('Loading...')
    
    # TODO: Evaluate if it is useful to include searching code in a function
    
    file_list = []
    # Verifying which type of task I've to classify (task or rest) 
    # and loads filename in different dirs
    for path_ in path_file_dirs:
        dir_list = [os.path.join(path_, f) for f in os.listdir(path_)]
        file_list = file_list + dir_list

    logger.debug('\n'.join(file_list))

    # Verifying which kind of analysis I've to perform (single or group) 
    # and filter list elements   
    if cmp(analysis, 'single') == 0:
        file_list = [elem for elem in file_list 
                     if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) ]#and (elem.find('mni') == -1)]
    else:
        file_list = [elem for elem in file_list 
                     if elem.find(img_pattern) != -1 and elem.find(task) != -1 and elem.find('mni') != -1]
    
    logger.debug('----------------- After filtering ------------------')
    logger.debug('\n'.join(file_list))
    
    # if no file are found I perform previous analysis!        
    if (len(file_list) <= runs and len(file_list) == 0):
        logger.error('Files not found, check the path of data!')
        raise ValueError()
    else:
        logger.debug('File corrected found ....')  

    ### Data loading ###
    file_list.sort()
    
    """
    image_list = []
    
    for img in file_list:
         
        if os.path.exists(os.path.join(path_file_dirs[0], img)):
            filepath = os.path.join(path_file_dirs[0], img)
        else:
            filepath = os.path.join(path_file_dirs[1], img)
    
        image_list.append(filepath)
    """        
    return file_list


def load_fmri(fname_list, skip_vols=0):
    """
    """   
    image_list = []
        
    for file_ in fname_list:
        
        logger.info('Now loading '+file_)     
        
        img = ni.load(file_)
    
        logger.debug(img.shape)
        if skip_vols != 0:
            
            data = img.get_data()
            img = img.__class__(data[:,:,:,skip_vols:], 
                                  affine = img.get_affine(), 
                                  header = img.get_header())
            del data
            
        image_list.append(img)
    
    logger.debug('The image list is of ' + str(len(image_list)) + ' images.')
    return image_list



      

def load_spatiotemporal_dataset(ds, **kwargs):
    
    onset = 0
    
    for arg in kwargs:
        if (arg == 'onset'):
            onset = kwargs[arg]
        if (arg == 'duration'):
            duration = kwargs[arg]
        if (arg == 'enable_results'):
            enable_results = kwargs[arg]
        
        
        
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)   
    
    #task_events = [e for e in events if e['targets'] in ['Vipassana','Samatha']]
    
    if 'duration' in locals():
        events = [e for e in events if e['duration'] >= duration]
    else:
        duration = np.min([ev['duration'] for ev in events])

    for e in events:
        e['onset'] += onset           
        e['duration'] = duration
        
    evds = eventrelated_dataset(ds, events = events)
    
    return evds


#@profile
def load_mask(path, subj, **kwargs):
    '''
    @param mask_type: indicates the type of atlas you want to use:
            - wu : Washington University ROIs extracted during experiments
            - juelich : FSL standard Juelich Maps.
    @param mask_area: mask is a string indicating the area to be analyzed
            - total: the entire brain voxels;
            - searchlight_3: mask from searchlight analysis exploited with radius equal to 3
            - searchlight_5: mask from searchlight analysis exploited with radius equal to 5
            - visual or other (broca, ba1 ecc.): other masks (with mask_type = 'wu' the only field could be 'visual')
            - v3a7 : Visual Cortex areas V3 V3a V7
            - intersect : intersection of v3a7 and searchlight 5
            - ll : Lower Left Visual Quadrant (it could be applied only with mask_type = 'wu')
            - lr : Lower Right Visual Quadrant (it could be applied only with mask_type = 'wu')
            - ul : Upper Left Visual Quadrant (it could be applied only with mask_type = 'wu')
            - ur : Upper Right Visual Quadrant (it could be applied only with mask_type = 'wu') 
    @param mask_space: Coordinate space of the mask used. (Needed only for Juelich Maps!)
            - mni: The standard MNI space. 
            - wu: The Washington University space.
    '''
    for arg in kwargs:
        if (arg == 'mask_atlas'):
            mask_type = kwargs[arg]
        if (arg == 'mask_area'):
            mask_area = kwargs[arg]  
        if (arg == 'mask_dir'):
            path = kwargs[arg]  
    
    if mask_area == 'none':
        return None        
            
    if (mask_type == 'wu'):
        mask = load_mask_wu(path, subj, **kwargs)
    else:
        if (mask_area == 'total'):
            mask = load_mask_wu(path, subj, **kwargs)
        else:
            mask = load_mask_juelich(**kwargs)
        
    return mask



def load_mask_wu(path, subj, **kwargs):
    

    mask_area = ['total']   
    roi_folder = '1_single_ROIs'
    isScaled = False
    sub_dir = ['none']
    
    for arg in kwargs:
        if (arg == 'mask_area'):
            mask_area = kwargs[arg].split(',')
        if (arg == 'roi_folder'):
            roi_folder = kwargs[arg]
        if (arg == 'coords'): #To be implemented
            coords = kwargs[arg]
        if (arg == 'sub_dir'):
            sub_dir = kwargs[arg].split(',')
        if (arg == 'scaled'):
            isScaled = kwargs[arg]
            

    '''
    for m in mask_list:
        if coords in locals():
            return
    '''
    mask_to_find = ''

    mask_path = os.path.join(path, roi_folder)
    
    scaled = ''
    
    logger.debug(mask_area)
    
    if isScaled == 'True':
        scaled = 'scaled'
    
    if (mask_area == ['visual']):
        mask_list = os.listdir(mask_path)
        mask_list = [m for m in mask_list if m.find(scaled) != -1 and m.find('hdr')!=-1 ]
                                 
    elif (mask_area == ['total']):
        #mask_path = os.path.join(path, subj)
        mask_path = os.path.join(path)
        mask_list = os.listdir(mask_path)
        mask_to_find1 = subj+'_mask_mask'
        mask_to_find2 = 'mask_'+subj+'_mask'
        mask_to_find3 = subj+'_mask.nii.gz'

        mask_list = [m for m in mask_list if m.find(mask_to_find1) != -1 \
                     or m.find(mask_to_find2) != -1 \
                     or m.find(mask_to_find3) != -1 \
                     or m.find('brain_mask') != -1]
        
    elif (mask_area == ['searchlight_3'] or mask_area == ['searchlight_5']):
        mask_list = os.listdir(mask_path)
        if mask_area == ['searchlight_3']:
            mask_to_find = 'mask_sl_r_3_t_54'
        else:
            mask_to_find = 'mask_sl_r_5_t_54'
        mask_list = [m for m in mask_list if m.find(mask_to_find) != -1]      
    else:
        mask_list_1 = os.listdir(mask_path)
        mask_list = []
        for m_ar in mask_area:
            mask_list = mask_list + [m for m in mask_list_1 if #(m.find('nii.gz') != -1 and m.find(m_ar)!=-1) or
                      ((m[:].find(m_ar) != -1 or m[-15:].find(m_ar) !=-1) and m.find('nii.gz') != -1) or
                      ((m[:].find(m_ar) != -1 or m[-15:].find(m_ar) !=-1) and m.find('hdr') != -1)]

    
    
    logger.debug(' '.join(mask_list))
    logger.info('Mask searched in '+mask_path+' Mask(s) found: '+str(len(mask_list)))
    
    files = []
    if len(mask_list) == 0:
        mask_path = os.path.join(path, subj)
        dir = sub_dir[0]
        if dir == 'none':
            dir = ''
        
        files = files + os.listdir(os.path.join(path, subj, dir))
            
        first = files.pop()
        
        bet_wu_data_(path, subj, dir)
    
        mask_list = [subj+'_mask_mask.nii.gz']
    
    
    data = 0
    for m in mask_list:
        img = ni.load(os.path.join(mask_path,m))
        data = data + img.get_data() 
        logger.info('Mask used: '+img.get_filename())

    mask = ni.Nifti1Image(data.squeeze(), img.get_affine())
        
    return mask

   
def load_mask_juelich(**kwargs):

    mask_space = 'wu'
    mask_area = ['total']
    mask_excluded = 'none'
    for arg in kwargs:
        if (arg == 'mask_area'):
            mask_area = kwargs[arg].split(',')
        if (arg == 'mask_space'):
            mask_space = kwargs[arg]
        if (arg == 'mask_excluded'):
            mask_excluded = kwargs[arg]   
            
            
    mask_path = os.path.join('/media/DATA/fmri/ROI_MNI',mask_space)

    mask_list_1 = os.listdir(mask_path)
    mask_list = []
    for m_ar in mask_area:
        mask_list = mask_list + [m for m in mask_list_1 if m.find(m_ar) != -1 
                                 and m.find(mask_excluded) == -1]
    data = 0
    
    for m in mask_list:
        img = ni.load(os.path.join(mask_path,m))
        data = data + img.get_data() 
        logger.info('Mask used: '+img.get_filename())

    mask = ni.Nifti1Image(data, img.get_affine())

    return mask    


def load_attributes (path, task, subj, **kwargs):
    ## Should return attr and a code to check if loading has been exploited #####
    
    #Default header struct
    header = ['targets', 'chunks']
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        if (arg == 'event_file'):
            event_file = kwargs[arg]
        if (arg == 'fidl_type'):
            fidl_type = int(kwargs[arg])
        if (arg == 'event_header'):
            header = kwargs[arg].split(',')

            if len(header) == 1:
                header = np.bool(header[0])
            
    completeDirs = []
    for dir in sub_dirs:
        if dir == 'none':
            dir = ''
        if dir[0] == '/':
            completeDirs.append(dir)
            
        completeDirs.append(os.path.join(path,subj,dir))
    
    completeDirs.append(path)
    completeDirs.append(os.path.join(path,subj))
    
    attrFiles = []
    logger.debug(completeDirs)
    for dir in completeDirs:
        attrFiles = attrFiles + os.listdir(dir)

    attrFiles = [f for f in attrFiles if f.find(event_file) != -1]
    logger.debug(attrFiles)
    if len(attrFiles) > 2:
        attrFiles = [f for f in attrFiles if f.find(subj) != -1]
        
    
    if len(attrFiles) == 0:
        logger.error(' *******       ERROR: No attribute file found!        *********')
        logger.error( ' ***** Check in '+str(completeDirs)+' ********')
        return None
    
    
    #txtAttr = [f for f in attrFiles if f.find('.txt') != -1]
    txtAttr = [f for f in attrFiles if f.find('.txt') != -1]
    
    attrFilename = ''
    if len(txtAttr) > 0:
        
        for dir in completeDirs:
            if (os.path.exists(os.path.join(dir, txtAttr[0]))):
                attrFilename = os.path.join(dir, txtAttr[0])               
       
    else:
        
        for dir in completeDirs:
            if (os.path.exists(os.path.join(dir, attrFiles[0]))):
                fidl_convert(os.path.join(dir, attrFiles[0]), os.path.join(dir, attrFiles[0][:-5]+'.txt'), type=fidl_type)
                attrFilename = os.path.join(dir, attrFiles[0][:-5]+'.txt')

    logger.debug(header)
    attr = SampleAttributes(attrFilename, header=header)
    return attr


def modify_conc_list(path, subj, conc_filelist, extension=''):
    """
    Function used to internally modify conc path, if remote directory is mounted at different
    location, the new mounting directory is passed as parameter.
    """
    import glob
    new_list = []
    
    for fl in conc_filelist:
        
        #Leave the first path part
        fl = fl[fl.find(subj):]
        logger.debug(fl)
        
        #Leave file extension
        fname, _, _ = fl.split('.')
        new_filename = os.path.join(path,fname)

        new_list += glob.glob(new_filename+'.*'+extension)

        logger.debug(fname)
        logger.debug(new_filename)
    
    logger.debug(new_list)
    
    del conc_filelist
    return new_list



def read_file(filename):
     
    filename_list = []
    with open(filename, 'r') as fileholder:
        for name in fileholder:
            filename_list.append(name[name.find('/'):-1])
    
    logger.debug(' '.join(filename_list))
        
    return filename_list



def read_conc(path, subj, conc_file_patt, sub_dir=['']):
    
    logger.debug(path)
    
    #First we look for the conc file in the task folder
    conc_file_list = []
    for dir_ in sub_dir:
        conc_path = os.path.join(path, subj, dir_)
        logger.debug(conc_path)
        if os.path.exists(conc_path):
            file_list = os.listdir(conc_path)
            logger.debug(conc_file_list)
            conc_file_list += [f for f in file_list if f.find('.conc') != -1 and f.find(conc_file_patt) != -1]
    
    logger.debug('.conc files in sub dirs: '+str(len(conc_file_list)))
    #Then we look in the subject directory
    if len(conc_file_list) == 0:
        conc_path = os.path.join(path, subj)
        file_list = os.listdir(conc_path)
        conc_file_list += [f for f in file_list \
                          if f.find('.conc') != -1 and f.find(conc_file_patt) != -1]
        logger.debug(' '.join(conc_file_list))
        logger.debug('.conc files in sub dirs: '+str(len(conc_file_list)))
    
    c_file = conc_file_list[0]
    
    #logger
    logger.debug(' '.join(conc_file_list))
    
    #Open and check conc file
    conc_file = open(os.path.join(conc_path, c_file), 'r')
    s = conc_file.readline()
    logger.debug(s)
    try:
        #conc file used to have first row with file number
        n_files = np.int(s.split(':')[1])
    except IndexError, _:
        logger.error('The conc file is not recognized.')
        return read_file(os.path.join(conc_path, c_file))
        
    logger.debug('Number of files in conc file is '+str(n_files))
    
    #Read conc file
    i = 0
    filename_list = []
    while i < n_files:
        name = conc_file.readline()
        #Find the path that did not correspond to local file namespace
        filename_list.append(name[name.find('/'):-1])
        i = i + 1
    
    conc_file.close()
    logger.debug('\n'.join(filename_list))
    return filename_list



def read_remote_configuration(path):
        
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path,'remote.conf'))
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            logger.debug(item)
    
    return dict(configuration) 

    logger.info('Reading remote config file '+os.path.join(path,'remote.conf'))



#@profile
def read_configuration (path, experiment, section):
    
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path,experiment))
    
    
    logger.info('Reading config file '+os.path.join(path,experiment))
    
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




def update_subdirs(conc_file_list, subj, **kwargs):
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        
    i = 0
    
    logger.debug('Old subdir '+kwargs['sub_dir'])
    
    for directory in conc_file_list:
        
        #Find the directory name
        s_dir = directory[directory.find(subj)+len(subj)+1:directory.rfind('/')]
        
        if s_dir in sub_dirs:
            continue
        elif sub_dirs[i].find('/') != -1 or i > len(sub_dirs):
            sub_dirs.append(s_dir)
            #i = i + 1          
        else:
            sub_dirs[i] = s_dir
        i = i + 1
        
    kwargs['sub_dir'] = ','.join(sub_dirs)
    logger.debug('New subdir '+kwargs['sub_dir'])
    return kwargs
            
            
            
def _find_file(path, subj, pattern):
    
    return []
