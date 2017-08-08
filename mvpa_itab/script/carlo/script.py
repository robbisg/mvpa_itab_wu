import os
import numpy as np


#path = '/root/robbis/DATA/fmri/carlo_ofp/'
#path = "/home/robbis/fmri/carlo_ofp/"
path_remote = "/home/robbis/fmri/data7/Carlo_OFP/"
path_remote = "/home/robbis/mount/meg_carlo/Carlo_OFP/"

subdir = "analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise/"

subjects = os.listdir(path_remote)
subjects = [s for s in subjects if s[0] == 's']
subjects.sort()

rm_cmd = ""

for s in subjects:
    #orig_dir = os.path.join(s, subdir)
    #command = "cp --parents %s* %s" % (os.path.join(s,subdir), path)
    #print command
    #os.system(command)
    
    

    path_subj = os.path.join(path, s, subdir)
    filelist = os.listdir(path_subj)
    filelist = [f for f in filelist if f.find('.ifh') != -1]
    filelist.sort()
    for f in filelist:
        in_file = os.path.join(path_subj, f)
        out_file = os.path.join(path_subj, f[:-9])
        cmd = 'nifti_4dfp -n '+in_file+' '+out_file
        os.system(cmd)
        rm_cmd += 'rm '+in_file+'\n'
        rm_cmd += 'rm '+in_file[:-4]+'.img\n'
        #print cmd
        #os.system(cmd)
    
    filetotal = [os.path.join(path_subj, f[:-9]+'.nii') for f in filelist]
    cmd = 'fslmerge -t '+os.path.join(path_subj, "residuals.nii.gz")+" "+" ".join(filetotal)
    os.system(cmd)
    
    conditionlist = [[l for l in f.split('_')[7]] for f in filelist]
    conditionlist = np.array(conditionlist)
    filename = "%s_condition_list.txt" % (s)
    #np.savetxt(os.path.join(path_subj, filename), conditionlist, dtype=np.str)
    np.savetxt(os.path.join(path_subj, filename), conditionlist, fmt="%s", delimiter=",")
    
print rm_cmd

# copy fidl event to wkpsy01
path_orig = "/home/robbis/mount/meg_carlo/Carlo_OFP/"
path_dest = "/home/robbis/mount/wkpsy01/carlo_ofp/" 
subdir = "analysis_SEP/DE_ASS_noHP/"
fname = "%s_eventfiles_DE_ASS_SINGLE.txt"

for s in subjects:
    sub_ = s[:4]+s[-6:]
    orig_dir = os.path.join(path_orig, s, subdir, fname % (sub_))
    dest_dir = os.path.join(path_dest, s, subdir)
    
    command = "cp %s %s" % (orig_dir, dest_dir)
    print command
    
    
### from fidl to pymvpa attributes
from mvpa_itab.utils import beta_attributes, enhance_attibutes_ofp
for s in subjects:
    sub_ = s[:4]+s[-6:]
    eventfile = os.path.join(path_dest, s, subdir, fname % (sub_))
    # fidl2txt
    output_beta = os.path.join(path_dest, s, "%s_eventfile_beta.txt" % (s))
    beta_attributes(eventfile, output_beta, nrun=8)
    enhance_attibutes_ofp(output_beta, current_header=['targets', 'chunks'])
    
    

## reordering files
path_dest = "/home/robbis/mount/wkpsy01/carlo_ofp/" 
subdir = "analysis_SEP/DE_ASS_noHP/"
niidir = "analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise/"
fname = "%s_eventfiles_DE_ASS_SINGLE.txt"

filepattern = "%s_DE_ASS_noHP_res_SINGLE_%s_mag_333_t88.nii"

for s in subjects:
    sub_ = s[:4]+s[-6:]
    eventfname = os.path.join(path_dest, s, subdir, fname % (sub_))
    eventfile = open(eventfname)
    
    eventlist = eventfile.readline().split()
    tr = float(eventlist[0])
    eventlist = eventlist[1:]
    
    filelist = [os.path.join(path_dest, s, niidir, filepattern % (sub_, event)) for event in eventlist]
    
    output_fname = os.path.join(path_dest, s, niidir, "residuals_sorted.nii.gz")
    command = "fslmerge -t %s %s" %(output_fname, " ".join(filelist))
    print command
    os.system(command)
    
    
    
    
    
    
    




    