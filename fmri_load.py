import nibabel as ni
import os
import utils
from mvpa2.suite import fmri_dataset, SampleAttributes, debug

def load_fmri_dataset_3d(path, name, attr_p):
    
    '''Deprecated'''
    
    if __debug__:
        debug.active += ['VERBOSE']
    
    print '********* fMRI Loading ************'
   
    pathFile = path+name+'/'
    listafile = os.listdir(pathFile);
    listafile.sort();
    
    
    imgList = [];
    for item in listafile:
        if item.find('.img') != -1:
            filePath = pathFile+item
            niftiImg = ni.load(filePath)
            imgList.append(niftiImg)
    
    
    mask_out = path+'mask_'+name+'_mask.nii.gz'
    mask = path+'mask_'+name+'.nii.gz'
    
    if os.access(mask_out, os.R_OK):
        print 'Mask file found...'
        mask = mask_out
    else:
        print 'Extracting mask...'
        #utils.maskExtractor(filePath, mask)
        mask = mask_out
    
    attr = SampleAttributes(attr_p)
    
    ds = fmri_dataset(imgList, targets = attr.targets, chunks = attr.chunks, mask = mask)
    
    return ds

def load_fmri_dataset_4d(path, name, attr_p, mask = None):
    
    
    nii = path+name
    
    attr = SampleAttributes(attr_p)
    
    ds = fmri_dataset(nii, targets = attr.targets, chunks = attr.chunks, mask = mask)
    
    return ds