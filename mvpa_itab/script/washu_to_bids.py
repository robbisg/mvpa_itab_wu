"""
 This is the specification for BIDS format
 The hierarchy of the file system is:
 
 remote_dir/bids_files/derivatives/
 
 under that dir it follows this schema:
 
 <pipeline-name>: the name of the pipeline used to build these files
 - partecipants.tsv : a list of subjects with other informations (sex, group, age)
 - pipeline_description.json : the description of the pipeline
 - sub-<subject-label> : subject personal subdir
 -- <image-type> : func | anat | dwi
 --- <source_file>_space-<space>[_res-<XxYxZ>][_variant-<label>]_<pipeline-type>.nii[.gz] : preproc | others
 --- <source_file>_space-<space>[_res-<XxYxZ>][_variant-<label>]_brain_mask.nii[.gz] : brain mask
 --- <source_file>[_space-<space>][_variant-<label>]_label-<label>_roi.nii.gz : single ROI label (PFC/FFA others)
 - group: group dir
 -- func
 --- <source_file>_space-<space>[_res-<XxYxZ>][_variant-<label>]_brain_mask.nii[.gz] : group brain mask
 --- <source_file>[_space-<space>][_variant-<label>]_label-<label>_roi.nii.gz : group ROI label (PFC/FFA others)
 - event_file.txt
 
"""

import os


def transform(configuration_file):
       
    
    
    
    return