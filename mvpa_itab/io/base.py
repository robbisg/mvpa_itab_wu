#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
import os
from mvpa2.suite import fmri_dataset, SampleAttributes
from mvpa2.suite import eventrelated_dataset
import logging
import numpy as np
import nibabel as ni
from mvpa2.datasets.eventrelated import find_events
from mvpa2.base.dataset import vstack

from mvpa_itab.io.utils import add_subdirs, build_pathnames
from mvpa_itab.preprocessing.pipelines import StandardPreprocessingPipeline
from mvpa_itab.io.configuration import read_configuration
from mvpa_itab.main_wu import detrend_dataset, normalize_dataset
#from memory_profiler import profile

logger = logging.getLogger(__name__) 



def load_dataset(path, subj, folder, **kwargs):
    ''' Load a 2d dataset given the image path, the subject and the main folder of 
    the data.

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
    
    skip_vols = 0
    roi_labels = dict()
    
    
    for arg in kwargs:
        if arg == 'skip_vols':              # no. of canceled volumes
            skip_vols = np.int(kwargs[arg])
        if arg == 'roi_labels':             # dictionary of mask {'mask_label': string}
            roi_labels = kwargs[arg]
    
    
    # Load the filename list        
    file_list = load_filelist(path, subj, folder, **kwargs)   

    
    # Load data
    try:
        fmri_list = load_fmri(file_list, skip_vols=skip_vols)
    except IOError, err:
        logger.error(err)
        return
    
       
    # Loading attributes
    attr = load_attributes(path, subj, folder, **kwargs)
    
    if (attr == None) and (len(file_list) == 0):
        return None            


    # Loading mask 
    mask = load_mask(path, **kwargs)
    roi_labels['brain'] = mask
    
    # Check roi_labels
    roi_labels = load_roi_labels(roi_labels)
          
    
    # Load the pymvpa dataset.    
    try:
        logger.info('Loading dataset...')
        
        ds = fmri_dataset(fmri_list, 
                          targets=attr.targets, 
                          chunks=attr.chunks, 
                          mask=mask,
                          add_fa=roi_labels)
        
        logger.debug('Dataset loaded...')
    except ValueError, e:
        logger.error(subj + ' *** ERROR: '+ str(e))
        del fmri_list
    
    
    # Add filename attributes for detrending purposes   
    ds = add_filename(ds, fmri_list)
    del fmri_list
    
    
    # Update Dataset attributes
    ds = add_events(ds)
    
    
    # Name added to do leave one subject out analysis
    ds = add_subjectname(ds, subj)
    
    
    # If the attribute file has more fields than chunks and targets    
    ds = add_attributes(ds, attr)
         
    
    return ds 



def add_filename(ds, fmri_list):
    
    f_list = []
    for i, img_ in enumerate(fmri_list):
        f_list += [i+1 for _ in range(img_.shape[-1])]
        
    # For each volume we store to which file it belongs to
    ds.sa['file'] = f_list
    
    return ds
    


def add_attributes(ds, attr):
    
    
    logger.debug(attr.keys())
    
    try:
        for k in attr.keys():
            ds.sa[k] = attr[k]
    except BaseException as err:        
        logger.error(str(err))
        
    return ds




def add_events(ds):
    
    ev_list = []
    events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
    for i in range(len(events)):
        duration = events[i]['duration']
        for _ in range(duration):
            ev_list.append(i + 1)
    
    ds.a['events'] = events # Update event field
    ds.sa['events_number'] = ev_list # Update event number
    
    return ds



def add_subjectname(ds, subj):
    """
    This function takes a string (the name of the subject) 
    and add it to the dataset for each element of the dataset
    
    Parameters
    ----------
    
    ds : pymvpa dataset
        the dataset where attribute should be added
        
    subj : string
        the name of the subject
        
    
    Returns
    -------
    ds : pymvpa dataset modified
    """
    
    ds.sa['name'] = [subj for _ in range(len(ds.sa.targets))]
    
    return ds


   
def load_filelist(path, name, folder, **kwargs):
    ''' Load file given filename, the 

    Parameters
    ----------
    path : string
       specification of filepath to load
    name : string
        subject name (in general it specifies a subfolder under path)
    folder : string
        subfolder under subject folder (in general is the experiment name)
    \*\*kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    file_list : string list
       list of strings indicating the file pathname
    '''
    
    img_pattern='.nii.gz'
    
    for arg in kwargs:
        if (arg == 'img_pattern'):
            img_pattern = kwargs[arg] 
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')



    file_list = build_pathnames(path, name, sub_dirs)
  
    # Filter list
    file_list = [elem for elem in file_list 
                 if (elem.find(img_pattern) != -1)]

    logger.debug(' Matching files ')
    logger.debug('\n'.join(file_list))
    
    file_list.sort()
    
    return file_list
  


def load_roi_labels(roi_labels):
    
    roi_labels_dict = {}
    if roi_labels != None:
        for label, img in roi_labels.items():
            if isinstance(img, str):
                roi_labels_dict[label] = ni.load(img)
            else:
                roi_labels_dict[label] = img
                
    return roi_labels_dict




def load_fmri(filelist, skip_vols=0):
    """
    """   
    image_list = []
        
    for file_ in filelist:
        
        logger.info('Now loading '+file_)     
        
        img = ni.load(file_)     
        data = img.get_data()
        
        
        if len(data.shape) == 4:
        
            img = img.__class__(data[:,:,:,skip_vols:], 
                                  affine = img.affine, 
                                  header = img.header)
        del data
        image_list.append(img)
        
        logger.debug(img.shape)
    
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



def load_mask(path, **kwargs):
    
    for arg in kwargs: 
        if (arg == 'mask_dir'):
            mask_path = kwargs[arg]
        if (arg == 'brain_mask'):
            rois = kwargs[arg].split(',')
                              
    mask_list = find_roi(mask_path, rois)
       
    # Load Nifti from list
    data = 0
    for m in mask_list:
        img = ni.load(os.path.join(mask_path, m))
        data = data + img.get_data() 
        logger.info('Mask used: '+img.get_filename())

    mask = ni.Nifti1Image(data.squeeze(), img.affine)
    logger.debug("Mask shape: "+str(mask.shape))
        
    return mask



def find_roi(path, rois):
    
    logger.debug(rois)
    
    found_rois = os.listdir(path)
    mask_list = []
    
    for roi in rois:
        mask_list += [m for m in found_rois if m.find(roi) != -1]
   
    mask_list = [m for m in mask_list if m.find(".nii.gz")!=-1 or m.find(".img")!=-1 or m.find(".nii") != -1]
    
    logger.debug(' '.join(mask_list))
    logger.info('Mask searched in '+path+' Mask(s) found: '+str(len(mask_list)))
    
    return mask_list
 
    


def load_attributes (path, subj, task,  **kwargs):
    ## Should return attr and a code to check if loading has been exploited #####
    
    #Default header struct
    header = ['targets', 'chunks']
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        if (arg == 'event_file'):
            event_file = kwargs[arg]
        if (arg == 'event_header'):
            header = kwargs[arg].split(',')
            # If it's one item is a boolean
            if len(header) == 1:
                header = np.bool(header[0])
            
    
    directory_list = add_subdirs(path, subj, sub_dirs)
    
    attribute_list = []
    
    logger.debug(directory_list)
    
    for d in directory_list:
        temp_list = os.listdir(d)
        attribute_list += [os.path.join(d,f) for f in temp_list if f.find(event_file) != -1]
        
        
    logger.debug(attribute_list)
    
    # Small check
    if len(attribute_list) > 2:
        attribute_list = [f for f in attribute_list if f.find(subj) != -1]
        
    
    if len(attribute_list) == 0:
        logger.error('ERROR: No attribute file found!')
        logger.error( 'Checked in '+str(directory_list))
        return None
    
    
    logger.debug(header)
    
    attr_fname = attribute_list[0]
    
    attr = SampleAttributes(attr_fname, header=header)
    return attr



def load_subjectwise_ds(path, 
                       subjects, 
                       conf_file, 
                       task, 
                       extra_sa=None,  
                       **kwargs):
    """
    extra_sa: dict or None, sample attributes added to the final dataset, they should be
    the same length as the subjects.
    
    subject: either a list of subjects or a csv file
    
    """
    
    conf = read_configuration(os.path.join(path, conf_file), task)
           
    conf.update(kwargs)
    logger.debug(conf)
    
    data_path = conf['data_path']
    
    
    if isinstance(subjects, str):        
        subjects, extra_sa = load_subject_file(subjects)
        
    
    logger.info('Merging subjects from '+data_path)
    
    for i, subj in enumerate(subjects):
        
        ds = load_dataset(data_path, subj, task, **conf)
        
        ds = detrend_dataset(ds, task, **conf)
        ds = normalize_dataset(ds, **conf)
        
        # add extra samples
        if extra_sa != None:
            for k, v in extra_sa.iteritems():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
        
        
        # First subject
        if i == 0:
            ds_merged = ds.copy()
        else:
            ds_merged = vstack((ds_merged, ds))
            ds_merged.a.update(ds.a)
            
        
        del ds

    return ds_merged, ['group'], conf




def load_subject_ds(conf_file, 
                    task, 
                    extra_sa=None,
                    prepro=StandardPreprocessingPipeline(), 
                    **kwargs):
    
    """
    This is identical to load_subjectwise_ds but we can
    specify a preprocessing pipeline to manage data
    
    """
    
    # TODO: conf file should include the full path
    conf = read_configuration(conf_file, task)
           
    conf.update(kwargs)
    logger.debug(conf)
    
    data_path = conf['data_path']
    
    # Subject file should be included in configuration
    subject_file = conf['subjects']
    subjects, extra_sa = load_subject_file(subject_file)
        
    logger.info('Merging %s subjects from %s' % (str(len(subjects)), data_path))
    
    for i, subj in enumerate(subjects):
        
        ds = load_dataset(data_path, subj, task, **conf)
        
        if ds == None:
            continue
        
        ds = prepro.transform(ds)
        
        # add extra samples
        if extra_sa != None:
            for k, v in extra_sa.iteritems():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
        
        
        # First subject
        if i == 0:
            ds_merged = ds.copy()
        else:
            ds_merged = vstack((ds_merged, ds))
            ds_merged.a.update(ds.a)
            
        
        del ds

    return ds_merged




def get_ds_data(ds, target_attribute='targets'):
    """
    Returns X and y data from pymvpa dataset
    """
    
    return ds.samples, ds.sa[target_attribute].value




def load_subject_file(fname):
    """
        File format example
        >>>
        subject,group,group_split
        s01_160112alefor,1,1
        s02_160216micbra,1,1
        >>
    """
    
    subject_array = np.genfromtxt(fname, 
                                  delimiter=',', 
                                  dtype=np.string_)
        
    subjects = subject_array[1:,0]
    # TODO: Check for extra_sa
    extra_sa = {a[0]:a[1:] for a in subject_array.T}
    
    return subjects, extra_sa

