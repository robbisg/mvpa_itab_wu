#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
# pylint: disable=maybe-no-member, method-hidden
import nibabel as ni
import os
from main_wu import *
from utils import *
import matplotlib.pyplot as plt
from fsl_wrapper import *
from mvpa2.suite import find_events, fmri_dataset, SampleAttributes
from mvpa2.suite import dataset_wizard, eventrelated_dataset, vstack
import cPickle as pickle
import logging
import time
#from memory_profiler import profile

def get_time():
    """Utility to format time used during results saving.
       
       Returns
       -------
        str : Datetime in format yymmdd_hhmmss
    """
    
    #Time acquisition for file name!
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


def load_conc_fmri_data(conc_file_list, el_vols = 0, **kwargs):
    """This function loads fmri data from a list of.
       
       Returns
       -------
        str : Datetime in format yymmdd_hhmmss
    """   
    imgList = []
        
    for file_ in conc_file_list:
        
        logging.info('Now loading '+file_)     
        
        im = ni.load(file_)
        data = im.get_data()
    
        logging.debug(data.shape)
    
        new_im = ni.Nifti1Image(data[:,:,:,el_vols:], 
                                affine = im.get_affine(), 
                                header = im.get_header())
        del data, im
        imgList.append(new_im)
    
    return imgList

#@profile    
def load_wu_file_list(path, name, task, el_vols=None, **kwargs):
    """
    returns imgList
    
    @param path: 
    @param name:
    @param task:
    @param el_vols: 
    """
    
    #What does it means analysis=single???
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
            path_file_dirs.append(os.path.join(path,name,dir_))

   
    logging.info('Loading...')
    
    file_list = []
    #Verifying which type of task I've to classify (task or rest) and loads filename in different dirs
    for path in path_file_dirs:
        file_list = file_list + os.listdir(path)

    logging.debug(' '.join(file_list))

    #Verifying which kind of analysis I've to perform (single or group) and filter list elements   
    if cmp(analysis, 'single') == 0:
        file_list = [elem for elem in file_list 
                     if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) ]#and (elem.find('mni') == -1)]
    else:
        file_list = [elem for elem in file_list 
                     if elem.find(img_pattern) != -1 and elem.find(task) != -1 and elem.find('mni') != -1]

    logging.debug(' '.join(file_list))
    
    #if no file are found I perform previous analysis!        
    if (len(file_list) <= runs and len(file_list) == 0):
        raise OSError('Files not found, please check if data exists in '+str(path_file_dirs))
    else:
        logging.debug('File corrected found ....')  

    ### Data loading ###
    file_list.sort()
    
    image_list = []
    
    for img in file_list:
         
        if os.path.exists(os.path.join(path_file_dirs[0], img)):
            filepath = os.path.join(path_file_dirs[0], img)
        else:
            filepath = os.path.join(path_file_dirs[1], img)
    
        image_list.append(filepath)
            
    return image_list

def load_fmri(fname_list, skip_vols=0):
    
    image_list = []
        
    for file_ in fname_list:
        
        logging.info('Now loading '+file_)     
        
        im = ni.load(file_)
        data = im.get_data()
    
        logging.debug(data.shape)
    
        new_im = im.__class__(data[:,:,:,skip_vols:], 
                                affine = im.get_affine(), 
                                header = im.get_header())
        del data, im
        image_list.append(new_im)
    
    logging.debug('The image list is of ' + str(len(image_list)) + ' images.')
    return image_list
    
    
    

def load_beta_dataset(path, subj, type, **kwargs):
    
    for arg in kwargs:
        if arg == 'runs':
            runs = np.int(kwargs[arg])

    dataset = []
    targets = []
    for r in range(runs):
        filename = 'avg_stats_for_regions_'+subj+'_'+str(runs)+'run_'+subj+'_run'+str(r+1)+'_vox.txt'
        f = open(os.path.join(path, subj, filename), 'r')
        run_data = read_beta_file(f, r, **kwargs)
        dataset.append(run_data[0])
        targets.append(run_data[1])
        
    dataset = np.vstack(dataset)
    targets = np.vstack(targets)
    
    ds = dataset_wizard(dataset, targets=targets[:,0], chunks=targets[:,1])
    
    return ds

def read_beta_file(f, run, time_begin=1, time_end=14, **kwargs):
    #print kwargs
    for arg in kwargs:
        #print arg
        if arg == 'mask_area':
            mask = kwargs[arg].split(',')
        if arg == 'time_begin':
            time_begin = np.int(kwargs[arg])
        if arg == 'time_end':
            time_end = np.int(kwargs[arg])
            
    if 'time_begin' not in locals():
        time_begin = 0
    if 'time_end' not in locals():
        time_end = 600   
        
    file_lines = f.readlines()
        
    data_flag = False
    data = []
    condition = []
    runs = []
    line_ctr = 0
    for line in file_lines:
        
        s_line = line.split()
        
        if len(s_line) == 0:
            data_flag = False
            line_ctr = 0
            continue
          
        #if line.find('TIMECOURSE : ') != -1:
        if s_line[0] == 'TIMECOURSE':
            cond, time_points = decode_line(line, 'timecourse')
            voxel_per_condition = 0
            
            interval = time_end - time_begin
            if interval >= time_points:
                interval = time_points
            if time_end > time_points:
                time_end = time_points
            if time_begin > time_points:
                time_begin = 0
            if time_begin >= time_end:
                interval = time_points
                
            for i in range(int(interval)):
                condition.append(cond)
                runs.append(run)
            
        #if line.find('REGION : ') !=-1:
        if s_line[0] == 'REGION':
            voxel_num, roi_name = decode_line(line, 'region')
            for area in mask:
                if roi_name.find(area) != -1:
                    voxel_per_condition += int(voxel_num)
                    data_flag = True
                    line_ctr = 0
           
        #print line
        if data_flag == True:
            #print line
            try:
                int(s_line[0])
            except ValueError, err:
                continue
            
            line_ctr += 1
            
            data_begin = 3 + time_begin
            data_end = data_begin +interval
            #print 'voxel_num='+str(voxel_num)
            #print 'line_ctr='+str(line_ctr)
            if line_ctr <= voxel_num:
                #print line.split()[data_begin:data_end]
                data.append(np.float_(np.array(line.split()[data_begin:data_end])))
    
    data_div = []
    ndata = np.array(data)
    for block in range(len(np.unique(condition))):
        data_div.append(ndata[block * voxel_per_condition:(block+1) * voxel_per_condition])
    
    ndata = np.hstack(data_div).T
    targets = np.vstack((condition, runs)).T
    
    return ndata, targets

def decode_line(line, keyword):
    
    if keyword == 'timecourse':
        
        condition = line.split()[2]
        time_points = line.split()[-1]
        return condition, time_points
    
    elif keyword == 'data':
        
        data = np.array(line)
        return data
    
    elif keyword == 'region':
        
        vox_num = line.split()[-1]
        roi_name = line.split()[-2]
        return vox_num, roi_name


def read_beta_data(line, voxel_num, time_points):
    

    return NotImplementedError()
    
    
#@profile
def load_dataset(path, subj, type_, **kwargs):
    '''
    @param mask: mask is a string indicating the area to be analyzed
            - total: the entire brain voxels;
            - visual: the visual cortex;
            - ll : Lower Left Visual Quadrant
            - lr : Lower Right Visual Quadrant
            - ul : Upper Left Visual Quadrant
            - ur : Upper Right Visual Quadrant
    '''
    
    use_conc = 'False'
    skip_vols = 0
    ext=''
    #print kwargs
    
    for arg in kwargs:
        if arg == 'skip_vols':
            skip_vols = np.int(kwargs[arg])
        if arg == 'use_conc':
            use_conc = kwargs[arg]
        if arg == 'conc_file':
            conc_file = kwargs[arg]
        if arg == 'sub_dir':
            sub_dir = kwargs[arg].split(',')
        if arg == 'img_extension':
            ext = kwargs[arg]
            
    if use_conc == 'False':
        file_list = load_wu_file_list(path, name=subj, task=type_, **kwargs)   
    else:
        file_list = read_conc(path, subj, conc_file, sub_dir=sub_dir)
        file_list = modify_conc_list(path, subj, file_list, extension=ext)

        kwargs = update_subdirs(file_list, subj, **kwargs)
        #print kwargs
    
    try:
        fmri_list = load_fmri(file_list, skip_vols=skip_vols)
    except IOError, err:
        logging.error(err)
        return 0
    
    ### Code to substitute   
    [code, attr] = load_attributes(path, type_, subj, **kwargs)        
    if code == 0:
        del fmri_list
        raise IOError('Attributes file not found')
        
    files = len(fmri_list)  
    
    #Loading mask 
    mask = load_mask(path, subj, **kwargs)        
       
    volSum = 0;
        
    for i in range(len(fmri_list)):
            
        volSum += fmri_list[i].shape[3]
    
    #Check attributes/dataset sample mismatches
    if volSum != len(attr.targets):
        logging.debug('volume number: '+str(volSum)+' targets: '+str(len(attr.targets)))
        del fmri_list
        logging.error(subj + ' *** ERROR: Attributes Length mismatches with fMRI volumes! ***')
        raise ValueError('Attributes Length mismatches with fMRI volumes!')       
    
    #Load the dataset.
    try:
        logging.info('Loading dataset...')
        ds = fmri_dataset(fmri_list, targets=attr.targets, chunks=attr.chunks, mask=mask) 
        logging.info('Dataset loaded...')
    except ValueError, e:
        logging.error(subj + ' *** ERROR: '+ str(e))
        del fmri_list
        return 0;
    
    #Update dataset attributes
    ev_list = []
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)
    for i in range(len(events)):
        duration = events[i]['duration']
        for j in range(duration):
            ev_list.append(i+1)
               
    ds.a['events'] = events
    ds.sa['events_number'] = ev_list
    ds.sa['name'] = [subj for i in range(len(ds.sa.chunks))]
    
    try:
        ds.sa['frame'] = attr.frame
        ds.sa['trial'] = attr.trial
    except BaseException, e:
        logging.error('Frame and Trial attributes not found.')
    
    
    #Inserted for searchlight proof!
    #ds.sa['block'] = np.int_(np.array([(i/14.) for i in range(len(ds.sa.chunks))])-5*ds.sa.chunks)
    
    f_list = []
    for i in range(files):
        for _ in range(fmri_list[i].shape[-1:][0]):
            f_list.append(i+1)

    ds.sa['file'] = f_list
    
    del fmri_list
    
    return ds    

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
    
    logging.debug(mask_area)
    
    if isScaled == 'True':
        scaled = 'scaled'
    
    if (mask_area == ['visual']):
        mask_list = os.listdir(mask_path)
        mask_list = [m for m in mask_list if m.find(scaled) != -1 and m.find('hdr')!=-1 ]
                                 
    elif (mask_area == ['total']):
        mask_path = os.path.join(path, subj)
        mask_list = os.listdir(mask_path)
        mask_to_find1 = subj+'_mask_mask'
        mask_to_find2 = 'mask_'+subj+'_mask'
        mask_to_find3 = '_mask.nii.gz'
        mask_list = [m for m in mask_list if m.find(mask_to_find1) != -1 or m.find(mask_to_find2) != -1 \
                     or m.find(mask_to_find3) != -1 or m.find('brain_mask') != -1]
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
      
    logging.debug(' '.join(mask_list))
    logging.info('Mask searched in '+mask_path+' Mask(s) found: '+str(len(mask_list)))
    
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
        logging.info('Mask used: '+img.get_filename())

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
        logging.info('Mask used: '+img.get_filename())

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
            
    completeDirs = []
    for dir in sub_dirs:
        if dir == 'none':
            dir = ''
        if dir.find('/') != -1:
            completeDirs.append(dir)
            
        completeDirs.append(os.path.join(path,subj,dir))
    
    completeDirs.append(path)
    completeDirs.append(os.path.join(path,subj))
    
    attrFiles = []
    for dir in completeDirs:
        attrFiles = attrFiles + os.listdir(dir)

    attrFiles = [f for f in attrFiles if f.find(event_file) != -1]
    #print attrFiles
    if len(attrFiles) > 2:
        attrFiles = [f for f in attrFiles if f.find(subj) != -1]
        
    
    if len(attrFiles) == 0:
        logging.error(' *******       ERROR: No attribute file found!        *********')
        logging.error( ' ***** Check in '+str(completeDirs)+' ********')
        return [0, None]
    
    
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

    logging.debug(header)
    attr = SampleAttributes(attrFilename, header=header)
    return [1, attr]


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
        logging.debug(fl)
        
        #Leave file extension
        fname, ext1, ext2 = fl.split('.')
        new_filename = os.path.join(path,fname)

        new_list += glob.glob(new_filename+'.*'+extension)

        logging.debug(fname)
        logging.debug(new_filename)
    
    logging.debug(new_list)
    
    del conc_filelist
    return new_list


def read_file(filename):
    
    
    filename_list = []
    with open(filename, 'r') as fileholder:
        for name in fileholder:
            filename_list.append(name[name.find('/'):-1])
    
    logging.debug(' '.join(filename_list))
        
    return filename_list


def read_conc(path, subj, conc_file_patt, sub_dir=['']):
    
    logging.debug(path)
    
    #First we look for the conc file in the task folder
    conc_file_list = []
    for dir_ in sub_dir:
        conc_path = os.path.join(path, subj, dir_)
        logging.debug(conc_path)
        if os.path.exists(conc_path):
            file_list = os.listdir(conc_path)
            logging.debug(conc_file_list)
            conc_file_list += [f for f in file_list if f.find('.conc') != -1 and f.find(conc_file_patt) != -1]
    
    logging.debug('.conc files in sub dirs: '+str(len(conc_file_list)))
    #Then we look in the subject directory
    if len(conc_file_list) == 0:
        conc_path = os.path.join(path, subj)
        file_list = os.listdir(conc_path)
        conc_file_list += [f for f in file_list \
                          if f.find('.conc') != -1 and f.find(conc_file_patt) != -1]
        logging.debug(' '.join(conc_file_list))
        logging.debug('.conc files in sub dirs: '+str(len(conc_file_list)))
    
    c_file = conc_file_list[0]
    
    #Logging
    logging.debug(' '.join(conc_file_list))
    
    #Open and check conc file
    conc_file = open(os.path.join(conc_path, c_file), 'r')
    s = conc_file.readline()
    logging.debug(s)
    try:
        #conc file used to have first row with file number
        n_files = np.int(s.split(':')[1])
    except IndexError, err:
        logging.error('The conc file is not recognized.')
        return read_file(os.path.join(conc_path, c_file))
        
    logging.debug('Number of files in conc file is '+str(n_files))
    
    
    #Read conc file
    i = 0
    filename_list = []
    while i < n_files:
        name = conc_file.readline()
        #Find the path that did not correspond to local file namespace
        filename_list.append(name[name.find('/'):-1])
        i = i + 1
    
    conc_file.close()
    logging.debug('\n'.join(filename_list))
    return filename_list


def read_remote_configuration(path):
        
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path,'remote.conf'))
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            logging.debug(item)
    
    return dict(configuration) 

    logging.info('Reading remote config file '+os.path.join(path,'remote.conf'))


#@profile
def read_configuration (path, experiment, section):
    
    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    
    config.read(os.path.join(path,experiment))
    
    
    logging.info('Reading config file '+os.path.join(path,experiment))
    
    types = config.get('path', 'types').split(',')
    
    if types.count(section) > 0:
        types.remove(section)
    
    for typ in types:
        config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            logging.debug(item)
    
    return dict(configuration)   


def save_results(path, results, configuration):

    datetime = get_time()
    analysis = configuration['analysis_type']
    mask = configuration['mask_area']
    task = configuration['analysis_task']
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    parent_dir = os.path.join(path, '0_results', new_dir)
    
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
        logging.info('Result saved in '+parent_dir)
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
    
    logging.info('Result saved in '+parent_dir)
    
    return 'OK' 

def save_results_searchlight (path, results):
    
    parent_dir = path 
    
    total_map = []
    
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        map = results[name]['map']
        
        radius = np.int(results[name]['radius'])
        
        
        if len(map.get_data().shape) > 3:
            mean_map = map.get_data().mean(axis=3)
            mean_img = ni.Nifti1Image(mean_map, affine=map.get_affine())
            fname = name+'_radius_'+str(radius)+'_searchlight_mean_map.nii.gz'
            ni.save(mean_img, os.path.join(results_dir,fname))
        else:
            mean_map = map.get_data()
            
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'
        ni.save(map, os.path.join(results_dir,fname))
        
        total_map.append(mean_map)
    
    total_map = np.array(total_map).mean(axis=0)
    total_img = ni.Nifti1Image(total_map, affine=map.get_affine())
    fname = 'accuracy_map_radius_'+str(radius)+'_searchlight_all_subj.nii.gz'
    ni.save(total_img, os.path.join(path,fname))
                   
    logging.info('Results writed in '+path)
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
    
    
    logging.info('Result saved in '+parent_dir)
    
    return 'OK' 


def save_results_transfer_learning(path, results):
    
    p = results[results.keys()[0]]['mahalanobis_similarity'][0].T[1]
    r = results[results.keys()[0]]['mahalanobis_similarity'][0].T[0]
    
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
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name)                        
        
        stats = results[name]['stats']
        fname = name+'_stats.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(stats))
        p_value = results[name]['p-value']
        file_.write('\n\n p-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        file_.write('\n\n Mean each fold p-value: '+str(p_value.mean()))
        file_.write('\n\n Mean null dist total accuracy value: '+str(results[name]['p']))
        file_.write('\nd-prime coefficient: '+str(results[name]['d_prime']))
        file_.write('\nbeta coefficient: '+str(results[name]['beta']))
        file_.write('\nc coefficient: '+str(results[name]['c']))
        #file.write('\n\nd-prime mahalanobis coeff: '+str(results[name]['d_prime_maha']))
        file_.close()
        
        if name == 'group':
            fname = name+'_fold_stats.txt'
            file_ = open(os.path.join(results_dir,fname), 'w')
            for m in stats.matrices:
                file_.write(str(m.stats['ACC']))
                file_.write('\n')
            file_.close()
                
        obj = results[name]['classifier'].ca
        fname = name+'_'+'classifier'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        
        obj = results[name]['targets']
        fname = name+'_'+'targets'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        
        obj = results[name]['predictions']
        fname = name+'_'+'predictions'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        #plot_transfer_graph(results_dir, name, results[name])
        
        c_m = results[name]['confusion_target']
        fname = name+'_confusion_target.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(c_m))
        file_.close()
        
        c_m = results[name]['confusion_total']
        fname = name+'_confusion_total.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(c_m))
        file_.close()        
        
        full_data = results[name]['mahalanobis_similarity'][0]
        true_pred = results[name]['mahalanobis_similarity'][1]
        threshold = results[name]['mahalanobis_similarity'][2]
        distances = results[name]['mahalanobis_similarity'][4]
        
        t_mahala = full_data[true_pred]
        fname = name+'_mahalanobis_data.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        n_src_label = len(np.unique(full_data.T[1]))
        n_tar_label = len(np.unique(full_data.T[0]))
        
        plot_list = dict()
        
        
        for t in np.unique(full_data.T[1]):
            f, ax = plt.subplots(2, 1)
            plot_list[t] = [f, ax]
        
        #for each label of the target dataset (target classes)
        for tar in np.unique(full_data.T[0]): 
            
            #Rest mask
            t_pred_mask = full_data.T[0] == tar
            t_m_data = full_data[t_pred_mask * true_pred]
            
            #for each predicted label
            for lab in np.unique(full_data.T[1]):
                
                histo_fname = name+'_histo_'+lab+'_'+tar+'_dist.txt'
                histo_p_fname = name+'_histo_'+lab+'_'+tar+'_p.txt'
                
                
                
                all_vec = full_data[(full_data.T[1] == lab) * t_pred_mask]
                
                if len(t_m_data) != 0:
                    m_maha = t_m_data.T[1] == lab
                    true_vec = t_m_data[m_maha]
                    num = len(true_vec)
                    mean_maha = np.mean(np.float_(true_vec.T[2]))
                    mean_p = np.mean(np.float_(true_vec.T[3]))
                else:
                    num = 0
                    mean_maha = np.mean(np.float_(all_vec.T[2]))
                    mean_p = np.mean(np.float_(all_vec.T[3]))
                
                tot_mean = np.mean(np.float_(all_vec.T[2]))
                tot_p = np.mean(np.float_(all_vec.T[3]))
                               
                file_.write(tar+' '+lab+' '+str(num)+' '+str(mean_maha)+' '+str(mean_p)+' '+str(tot_mean)+' '+str(tot_p)+'\n')
                
                np.savetxt(os.path.join(results_dir,histo_fname), np.float_(all_vec.T[2]))
                np.savetxt(os.path.join(results_dir,histo_p_fname), np.float_(all_vec.T[3]))
                
                #bin_d = np.linspace(mn_d, mx_d, 25)
                #bin_p = np.linspace(0, 1, 25)
                
                plot_list[lab][1][0].hist(np.float_(all_vec.T[2]), bins=35, label=tar, alpha=0.5)
                plot_list[lab][1][1].hist(np.float_(all_vec.T[3]), bins=35, label=tar, alpha=0.5)          
                
                hist_sum[tar][lab]['dist'].append(np.float_(all_vec.T[2]))
                hist_sum[tar][lab]['p'].append(np.float_(all_vec.T[3]))
                
            
            histo_full_fname = name+'_'+tar+'_histo_.txt'
            histo_full_p_fname = name+'_'+tar+'_histo_p.txt'
            
            l = 0
            pred_array = np.zeros(full_data.T[1].shape)
            for lab in np.unique(full_data[t_pred_mask].T[1]):
                
                pred_array[full_data.T[1] == lab] = l
                
                l = l + 1
                
            #np.savetxt(os.path.join(results_dir,histo_full_fname), np.vstack((full_data[t_pred_mask].T[2:], pred_array)))


        for c in distances.keys():
            for tar in np.unique(full_data.T[0]):
                data = distances[c][full_data.T[0] == tar]
                f_d = plt.figure()
                a_d = f_d.add_subplot(111)
                a_d.plot(data)
                a_d.set_ylim(0, 75)
                step = data.__len__() / 6.
                for j in np.arange(6)+1:#n_runs
                    a_d.axvline(x = step * j, ymax=a_d.get_ylim()[1], color='y', linestyle='-', linewidth=1)
                a_d.axhline(y = threshold, color='r', linestyle='--', linewidth=2)
                
                means_s[c+'_'+tar].append(np.mean(data))
                a_d.axhline(y = np.mean(data), color='black', linestyle=':', linewidth=2)
                f_d.savefig(os.path.join(results_dir,name+'_distance_plot_'+c+'_'+tar+'_.png'))
            
                np.savetxt(os.path.join(results_dir,name+'_distance_txt_'+c+'_'+tar+'_.txt'), 
                           distances[c][full_data.T[0] == tar], fmt='%.4f')               

        
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
        fname = name+'_confusion_mahala.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            print err
        '''
        cmatrix_mahala = results[name]['confusion_tot_maha']
        fname = name+'_confusion_total_mahala.txt'
        file = open(os.path.join(results_dir,fname), 'w')
        try:
            file.write(str(cmatrix_mahala))
        except ValueError,err:
            file.write('None')
            print err  
        '''
        
        file_.close()
        
    
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
    
    logging.debug('Old subdir '+kwargs['sub_dir'])
    
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
    logging.debug('New subdir '+kwargs['sub_dir'])
    return kwargs
            
def _find_file(path, subj, pattern):
    
    return []
