#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
# pylint: disable=maybe-no-member, method-hidden
import nibabel as ni
import numpy as np
import os
from ..main_wu import *
from ..utils import *
import matplotlib.pyplot as plt
from ..fsl_wrapper import *
from mvpa2.suite import find_events, fmri_dataset, SampleAttributes
from mvpa2.suite import dataset_wizard, eventrelated_dataset, vstack
import cPickle as pickle
import logging
import time
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
    
    for arg in kwargs:
        if (arg == 'mask_area'):
            mask_area = kwargs[arg].split(',')
  
    #Mask issues    
    roi_folder = '1_single_ROIs'
    isScaled = False
    sub_dir = ['none']
    
    for arg in kwargs:
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
        fname, ext1, ext2 = fl.split('.')
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
    except IndexError, err:
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


def save_results(path, results, configuration):
    """
    path: is the results path dir
    results: is the structure used to store data
    configuration: is the configuration file to store analysis info
    """
    
    # Get information to make the results dir
    datetime = get_time()
    analysis = configuration['analysis_type']
    mask = configuration['mask_area']
    task = configuration['analysis_task']
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    # New directory to store files
    parent_dir = os.path.join(path, '0_results', new_dir)
    
    # Check which analysis has been done
    ## TODO: Use a class to save results
    if analysis == 'searchlight':
        save_results_searchlight(parent_dir, results)
    elif analysis == 'transfer_learning':
        save_results_transfer_learning(parent_dir, results)
        #write_all_subjects_map(path, new_dir)
    elif analysis == 'clustering':
        save_results_clustering(parent_dir, results)
    else:
        save_results_basic(parent_dir, results)
        write_all_subjects_map(path, new_dir)  

    ###################################################################        
    import csv
    w = csv.writer(open(os.path.join(parent_dir, 'configuration.csv'), "w"))
    for key, val in configuration.items():
        w.writerow([key, val])
        
    #######################################################################
    
    if analysis == 'searchlight':
        logger.info('Result saved in '+parent_dir)
        return 'OK' #WTF?
    else:
        file_summary = open(os.path.join(parent_dir, 'analysis_summary_'+mask+'_'+task+'.txt'), "w")
        res_list = []
    
        for subj in results.keys():
        
            list = [subj]
            file_summary.write(subj+'\t')
            
            r = results[subj]
        
            list.append(r['stats'].stats['ACC'])
            list.append(r['p'])
            
            file_summary.write(str(r['stats'].stats['ACC'])+'\t')
            file_summary.write(str(r['p'])+'\t')
            
            if analysis == 'transfer_learning':
                list.append(r['confusion_total'].stats['ACC'])
                list.append(r['d_prime'])
                list.append(r['beta'])
                list.append(r['c'])
                
                file_summary.write(str(r['confusion_total'].stats['ACC'])+'\t')
                file_summary.write(str(r['d_prime'])+'\t')
                file_summary.write(str(r['beta'])+'\t')
                file_summary.write(str(r['c'])+'\t')
                
            file_summary.write('\n')
            
        res_list.append(list)
        
        file_summary.close()
    
    logger.info('Result saved in '+parent_dir)
    
    return 'OK' 

def save_results_searchlight (path, results):
    
    parent_dir = path 
    
    total_map = []
    
    # For each subject save map and average across fold
    for name in results:
        
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        map_ = results[name]['map']
        
        radius = np.int(results[name]['radius'])
        
        
        if len(map_.get_data().shape) > 3:
            mean_map = map_.get_data().mean(axis=3)
            mean_img = ni.Nifti1Image(mean_map, affine=map_.get_affine())
            fname = name+'_radius_'+str(radius)+'_searchlight_mean_map.nii.gz'
            ni.save(mean_img, os.path.join(results_dir,fname))
        else:
            mean_map = map_.get_data()
        
        
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'
        ni.save(map_, os.path.join(results_dir,fname))
        
        total_map.append(mean_map)
    
    # Save the total average map
    total_map = np.array(total_map).mean(axis=0)
    total_img = ni.Nifti1Image(total_map, affine=map_.get_affine())
    fname = 'accuracy_map_radius_'+str(radius)+'_searchlight_all_subj.nii.gz'
    ni.save(total_img, os.path.join(path,fname))
                   
    logger.info('Results writed in '+path)
    return path

def save_results_basic(path, results):
    
    parent_dir = path 
    
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        for key in results[name]:
            
            if key == 'map':
                
                m_mean = results[name]['map'].pop()
                m_mean_data = m_mean.get_data()
                fname = name+'_mean_map.nii.gz'
                m_mean_data = (m_mean_data - np.mean(m_mean_data))/np.std(m_mean_data)
                
                m_mean_zscore = ni.Nifti1Image(m_mean_data, m_mean.get_affine())
                ni.save(m_mean_zscore, os.path.join(results_dir,fname))
                
                #save_map(os.path.join(results_dir, fname), m_mean_data, m_mean.get_affine())
                
                for map, t in zip(results[name][key], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
                    map_data = map.get_data()
                    map_data_zscore = (map_data - np.mean(map_data))/np.std(map_data)
                    map_zscore = ni.Nifti1Image(map_data_zscore, map.get_affine())
                    ni.save(map_zscore, os.path.join(results_dir,fname))
                    
            elif key == 'stats':
                stats = results[name][key]
                fname = name+'_stats.txt'
                file = open(os.path.join(results_dir,fname), 'w')
                file.write(str(stats))
                file.close()
            else:
                obj = results[name][key]
                if key == 'classifier':
                    obj = results[name][key].ca
                fname = name+'_'+key+'.pyobj'          
                file = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file)
                file.close()
     
    ###################################################################          
    
    
    logger.info('Result saved in '+parent_dir)
    
    return 'OK' 


def save_results_transfer_learning(path, results):
    
    # Cross-decoding predictions and labels
    # p = classifier prediction on target ds
    # r = targets of target ds
    p = results[results.keys()[0]]['mahalanobis_similarity'][0].T[1]
    r = results[results.keys()[0]]['mahalanobis_similarity'][0].T[0]
    
    # Stuff for total histograms 
    hist_sum = dict()
    
    means_s = dict()
    for t in np.unique(r):
        hist_sum[t] = dict()
        for l in np.unique(p):
            hist_sum[t][l] = dict()
            hist_sum[t][l]['dist'] = []
            hist_sum[t][l]['p'] = []
            means_s[l+'_'+t] = []
    
    
    for name in results:
        # Make subject dir
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name)                        
        
        # Statistics of decoding
        stats = results[name]['stats']
        fname = name+'_stats.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        file_.write(str(stats))
        
        p_value = results[name]['pvalue']
        file_.write('\n\n p-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        
        file_.write('\n\n Mean each fold p-value: '+str(p_value.mean()))
        file_.write('\n\n Mean null dist total accuracy value: '+str(results[name]['p']))
        file_.write('\nd-prime coefficient: '+str(results[name]['d_prime']))
        file_.write('\nbeta coefficient: '+str(results[name]['beta']))
        file_.write('\nc coefficient: '+str(results[name]['lab']))
        #file.write('\n\nd-prime mahalanobis coeff: '+str(results[name]['d_prime_maha']))
        file_.close()
        
        if name == 'group':
            fname = name+'_fold_stats.txt'
            file_ = open(os.path.join(results_dir,fname), 'w')
            for m in stats.matrices:
                file_.write(str(m.stats['ACC']))
                file_.write('\n')
            file_.close()
        
        
        for k in results[name].keys():
            if k in ['classifier', 'targets', 'predictions']:
                if k == 'classifier':
                    obj = results[name][k].ca
                else:
                    obj = results[name][k]
                    
                fname = name+'_'+k+'.pyobj'
        
                file_ = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file_)
                file_.close()
        
            if k in ['confusion_target', 'confusion_total']:
                c_m = results[name][k]
                fname = name+'_'+k+'.txt'
                file_ = open(os.path.join(results_dir,fname), 'w')
                file_.write(str(c_m))
                file_.close()
                
                
        #plot_transfer_graph(results_dir, name, results[name])      
        
        ####################### Similarity results #####################################
        # TODO: Keep in mind! They should be saved when similarity has been performed!!!
        full_data = results[name]['mahalanobis_similarity'][0]
        similarity_mask = results[name]['mahalanobis_similarity'][1]
        threshold = results[name]['mahalanobis_similarity'][2]
        distances = results[name]['mahalanobis_similarity'][4]
        
        # Renaming variables to better read
        ds_targets = full_data.T[0]
        class_prediction_tar = full_data.T[1]
        
        t_mahala = full_data[similarity_mask]
        fname = name+'_mahalanobis_data.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        n_src_label = len(np.unique(class_prediction_tar))
        n_tar_label = len(np.unique(ds_targets))
        
        plot_list = dict()
        
        # For each label in predictions
        for t in np.unique(class_prediction_tar):
            f, ax = plt.subplots(2, 1)
            plot_list[t] = [f, ax]
        

        
        
        # For each label of the target dataset (target classes)
        for tar in np.unique(ds_targets): 
            
            # Select data belonging to target loop class
            target_mask = ds_targets == tar
            similarity_target = similarity_mask[target_mask]
            target_data = full_data[target_mask] 


            for lab in np.unique(class_prediction_tar):
 
                # Select target data classified as loop label
                prediction_mask = target_data.T[1] == lab 
                crossd_data = target_data[prediction_mask] 
                similarity_crossd = similarity_target[prediction_mask]

                distance_ = crossd_data.T[2]
                p_values_ = crossd_data.T[3]                
                
                # Filter data that meets similarity criterion
                similarity_data = crossd_data[similarity_crossd]
                num = len(similarity_data)
                
                
                if len(target_data) != 0:
                    distance_data = similarity_data.T[2]# Mean distance
                    p_data = similarity_data.T[3] # Mean p-value
                else:
                    distance_data = np.mean(np.float_(distance_)) 
                    p_data = np.mean(np.float_(p_values_)) # 
                
                mean_maha_d = np.mean(np.float_(distance_data))
                mean_maha_p = np.mean(np.float_(p_data))
                
                tot_d = np.mean(np.float_(distance_))
                tot_p = np.mean(np.float_(p_values_))
                
                occurence_ = ','.join([tar, lab, str(num), str(mean_maha_d),str(mean_maha_p),str(tot_d),str(tot_p)])
                
                file_.write(occurence_+'\n')
                
                # TODO: Maybe is possible to collapse both file in a single one!
                histo_d_fname = "%s_hist_%s_%s_dist.txt" % (name, lab, tar)
                histo_p_fname = "%s_hist_%s_%s_p.txt" % (name, lab, tar)
                
                np.savetxt(os.path.join(results_dir,histo_d_fname), np.float_(distance_))
                np.savetxt(os.path.join(results_dir,histo_p_fname), np.float_(p_values_))
                
                # Histogram plots
                # TODO: Maybe it's better to do something else!
                # TODO: Unique values of bins!
                plot_list[lab][1][0].hist(np.float_(distance_), bins=35, label=tar, alpha=0.5)
                plot_list[lab][1][1].hist(np.float_(p_values_), bins=35, label=tar, alpha=0.5)          
                
                # We store information for the total histogram
                hist_sum[tar][lab]['dist'].append(np.float_(distance_))
                hist_sum[tar][lab]['p'].append(np.float_(p_values_))
                
                
                
                ## TODO: Insert plot in a function and let the user decide if he wants it!
                ## plot_distances(distances, runs)
                ## distance_ = data[prediction_mask] # Equivalent form!
                data = distances[lab][target_mask]
                f_d = plt.figure()
                a_d = f_d.add_subplot(111)
                a_d.plot(data)
                a_d.set_ylim(data.mean()-3*data.std(), data.mean()+3*data.std())
                step = data.__len__() / 6.
                for j in np.arange(6)+1:#n_runs
                    a_d.axvline(x = step * j, ymax=a_d.get_ylim()[1], color='y', linestyle='-', linewidth=1)
                a_d.axhline(y = threshold, color='r', linestyle='--', linewidth=2)
                
                
                a_d.axhline(y = np.mean(data), color='black', linestyle=':', linewidth=2)
                
                pname = "%s_distance_plot_%s_%s.png" % (name, lab, tar)
                
                f_d.savefig(os.path.join(results_dir, pname))
                
                ## Save file ##
                fname = "%s_distance_txt_%s_%s.txt" % (name, lab, tar)
                np.savetxt(os.path.join(results_dir, fname), 
                           data, fmt='%.4f')               

                means_s[lab+'_'+tar].append(np.mean(data))
                
        ## TODO: Insert in a function        
        for k in plot_list.keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            fig_name = os.path.join(results_dir,name+'_histogram_'+k+'.png')
            plot_list[k][0].savefig(fig_name)

        file_.write('\nthreshold '+str(threshold))       
        file_.close()
        
        cmatrix_mahala = results[name]['confusion_mahala']
        fname = "%s_confusion_mahala.txt" % name
        file_ = open(os.path.join(results_dir,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            print err
        
        file_.close()
        
        # Is this snippert a part of other code???
        if results[name]['map'] != None:
            
            m_mean = results[name]['map'].pop()
            fname = name+'_mean_map.nii.gz'
            mask_ = m_mean._data != 0
            m_mean._data[mask_ ] = (m_mean._data[mask_ ] - np.mean(m_mean._data[mask_ ]))/np.std(m_mean._data[mask_ ])
            ni.save(m_mean, os.path.join(results_dir,fname))
        
            for map, t in zip(results[name]['map'], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
    
                    #map._data = (map._data - np.mean(map._data))/np.std(map._data)
                    map._dataobj = (map._dataobj - np.mean(map._dataobj))/np.std(map._dataobj)
                    ni.save(map, os.path.join(results_dir,fname))
    
    
    '''
    End of main for
    '''
                
    from pprint import pprint
    filename_ = open(os.path.join(path, 'distance_means.txt'), 'w')
    pprint(means_s, filename_)
    filename_.close()
    
    plot_list = dict()
    for t in hist_sum[hist_sum.keys()[0]].keys():
        f, ax = plt.subplots(2, 1)
        plot_list[t] = [f, ax]
    
    for k1 in hist_sum:
        
        for k2 in hist_sum[k1]:
            
            for measure in hist_sum[k1][k2]:
                hist_sum[k1][k2][measure] = np.hstack(hist_sum[k1][k2][measure])
                
        
            plot_list[k2][1][0].hist(hist_sum[k1][k2]['dist'], bins=35, label = k1, alpha=0.5)
            plot_list[k2][1][1].hist(hist_sum[k1][k2]['p'], bins=35, label = k1, alpha=0.5)
   
    for k in hist_sum[hist_sum.keys()[0]].keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            plot_list[k][0].savefig(os.path.join(path,'total_histogram_'+k+'.png'))
    
    plt.close('all')    
    return


def save_results_clustering(path, results):
    
    
    for name in results:
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name) 
        
        for key in results[name]:
            obj = results[name][key]
            fname = name+'_'+key+'.pyobj'          
            file = open(os.path.join(results_dir,fname), 'w')
            pickle.dump(obj, file)
            file.close()
            
            if key == 'clusters':
                plot_clusters_graph(results_dir, name, results[name])
  
    return

def save_map(filename, map_np_array, affine=np.eye(4)):
        
    map_zscore = ni.Nifti1Image(map_np_array, affine)
    ni.save(map_zscore, filename)
    return

def write_all_subjects_map(path, dir_):
    
    res_path = os.path.join(path, '0_results', dir_)
    
    subjects = os.listdir(res_path)
    subjects = [s for s in subjects if s.find('.') == -1]
    
    img_list = []
    i = 0
    list_t = []
    
    min_t = 1000
    max_t = 0
    
    #Get data from path
    for s in subjects:
        
        s_path = os.path.join(res_path, s)
        map_list = os.listdir(s_path)
        
        map_list = [m for m in map_list if m.find('mean_map') != -1]
        
        img = ni.load(os.path.join(s_path, map_list[0]))
        
        #Store maximum and minimum if maps are dimension mismatching
        
        if img.get_data().shape[-1] < min_t:
            min_t = img.get_data().shape[-1]
        
        if img.get_data().shape[-1] > max_t:
            max_t = img.get_data().shape[-1]
            
        img_list.append(img)
        list_t.append(img.get_data().shape[-1])
        i = i + 1
        
    dim_mask = np.array(list_t) == min_t
    img_list_ = np.array(img_list)
    
    #compensate for dimension mismatching
    n_list = []
    if dim_mask.all() == False:
        img_m = img_list_[dim_mask]
        n_list = []
        for img in img_m:
            data = img.get_data()
            for i in range(max_t - data.shape[-1]):
                add_img = np.mean(data, axis=len(data.shape)-1)
                add_img =  np.expand_dims(add_img, axis=len(data.shape)-1)
                data = np.concatenate((data, add_img), axis=len(data.shape)-1)
            n_list.append(data)   
        v1 = np.expand_dims(img_list_[~dim_mask][0].get_data(), axis=0)
    else:
        v1 = np.expand_dims(img_list_[dim_mask][0].get_data(), axis=0)
    
    for img in img_list_[~dim_mask][1:]:
        v1 = np.vstack((v1,np.expand_dims(img.get_data(), axis=0)))
    
    for img_data in n_list:
        v1 = np.vstack((v1,np.expand_dims(img_data, axis=0)))
        
    m_img = np.mean(v1, axis=0)
    
    img_ni = ni.Nifti1Image(m_img.squeeze(), img.get_affine())
    
    filename = os.path.join(res_path, 'all_subjects_mean_map_prova.nii.gz')
    ni.save(img_ni, filename)
    
    return


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
