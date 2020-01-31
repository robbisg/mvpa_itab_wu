import os
import nibabel as ni
import numpy as np
import logging
logger = logging.getLogger(__name__) 


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
    except IndexError:
        logger.error('The conc file is not recognized.')
        return read_file(os.path.join(conc_path, c_file))
        
    logger.debug('Number of files in conc file is %s', str(n_files))
    
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


def update_subdirs(conc_file_list, subj, **kwargs):
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        
    
    logger.debug('Old subdir '+kwargs['sub_dir'])
    
    for i, directory in enumerate(conc_file_list):
        
        #Find the directory name
        s_dir = directory[directory.find(subj)+len(subj)+1:directory.rfind('/')]
        
        if s_dir in sub_dirs:
            continue
        elif sub_dirs[i].find('/') != -1 or i > len(sub_dirs):
            sub_dirs.append(s_dir)       
        else:
            sub_dirs[i] = s_dir
        
    kwargs['sub_dir'] = ','.join(sub_dirs)
    logger.debug('New subdir '+kwargs['sub_dir'])
    return kwargs