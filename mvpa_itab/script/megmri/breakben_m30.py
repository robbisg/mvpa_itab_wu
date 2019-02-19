import nibabel as ni
import numpy as np
import os


# Load Atlas

atlases = dict()
atlases['juelich'] = ni.load("/usr/share/fsl/data/atlases/Juelich/Juelich-maxprob-thr25-1mm.nii.gz")
atlases['oxford'] = ni.load("/usr/share/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")

# Extract masks

rois = {
        "omega": {"juelich": [49,50,47,58]},
        "callosum_body" : {"juelich": [95]},
        "basal_ganglia": {"oxford": [15,16,4,5]},
        "insula": {"juelich":[116,117,118,119,120,121]}
        }


# Save masks
total_mask = []
mask_paths = []
for i, (area, values) in enumerate(rois.items()):
    for atlas, value in values.items():
        
        data = atlases[atlas].get_data()
        affine = atlases[atlas].affine
        mask = np.zeros_like(data)
        
        for v in value:
            mask = np.logical_or(mask, data == v)
            
        
    img = ni.Nifti1Image(np.int_(mask), affine)
    mask_paths.append("/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/%s.nii.gz" %(area))
    #ni.save(img, "/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/%s.nii.gz" %(area))
    total_mask.append(np.int_(mask)*(i+1))
            
mask_paths = ["/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze/%s.nii.gz" %(area) for area in rois.keys()]

mask_paths.append("/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze/intracranial_rois.nii.gz")

# Flirt T2 to HF.lres




# Flirt atlas to HF.lres
path = "/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze"
command = "convert_xfm -omat %s/mni2lanl.mat -inverse %s/lanl_to_mni.mat" %(path, path)
print command

command = "convert_xfm -omat %s/mni2lflanl.mat -concat %s/hf2lf.mat %s/mni2lanl.mat" %(path, path, path)
print command


mat_file = "/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze/mni2lflanl.mat"
ref = "/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze/lanl_hf_lres_reg_orient.nii.gz"

new_masks = []
for mask in mask_paths:
    output = mask[:-7]+"_lanl.nii.gz"
    command = "flirt -in %s -ref %s -applyxfm -init %s -o %s -interp nearestneighbour" %(
                                                                mask,
                                                                ref,
                                                                mat_file,
                                                                output
                                                                )
    
    new_masks.append(output)
    
    print command




# Conversion

path = "/home/robbis/phantoms/Coregistration/BREAKBEN/lanl/"

list_ = os.listdir(path)
list_ = [f for f in list_ if f.find(".nii.gz") != -1]

for f in list_:
    
    command = "fslorient -deleteorient %s" %(os.path.join(path, f))
    print command
    
    command = "fslswapdim %s z -x y %s" %(os.path.join(path, f), os.path.join(path, 'converted', f))
    print command
    
    command = "fslorient -setqformcode 1 %s" %(os.path.join(path, 'converted', f))
    print command
