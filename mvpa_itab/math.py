import nibabel as ni
import numpy as np
import os


def remove_value(image, value, mask=None):
    
    if not isinstance(mask, np.ndarray):
        mask = np.ones_like(image)
    elif len(mask.shape) != len(image.shape):
        mask = mask[...,np.newaxis]
        
    mask_ = mask / mask
    mask_[np.isnan(mask_)] = 0
        
    image -= (mask_ * value)
    
    return image
    
    
    

def remove_value_nifti(img_fname, value, output_fname, mask_fname=None):
    
    img = ni.load(img_fname)
    
    if mask_fname != None:
        mask = ni.load(mask_fname)
    
    out_img = remove_value(img.get_data(), value, mask.get_data())
    
    ni.save(ni.Nifti1Image(out_img, img.affine), output_fname)
    
    return out_img



def remove_mean_nifti(img_fname, output_fname, mask_fname=None):
    
    img = ni.load(img_fname)
    
    data = img.get_data()
    
    value = data.mean()
    
    mask_data = None
    if mask_fname != None:
        mask_data = ni.load(mask_fname).get_data()
        value = data[np.bool_(mask_data)].mean()
        print value
        
     
    out_img = remove_value(data, value, mask_data)
    
    ni.save(ni.Nifti1Image(out_img, img.affine), output_fname)
    
    return out_img



def conjunction_map(image_map, mask, output_fname, output='mask'):
    
    """
    if output is mask a mask composed by all ones is the output
    else a mask with image_map values in nonzero mask voxels
    """
    
    img = ni.load(image_map)
    mask_img = ni.load(mask)
    
    mask_int = np.int_(mask_img.get_data() != 0)
    
    data = img.get_data()
    if output == 'mask':
        data /= data
    
    out_img = np.float_(data * mask_int)
        
    ni.save(ni.Nifti1Image(out_img, img.affine), output_fname)
    
    return



def afni_converter(afni_fname, output_fname, brick):
    
    command = "3dTcat -prefix %s %s[%s]" % (output_fname, afni_fname, str(brick))
    print(command)
    os.system(command)
    
    img = ni.load(output_fname)
    output = img.get_data().squeeze()
    
    ni.save(ni.Nifti1Image(output, img.affine), output_fname)
    
    return
    
    
    
    