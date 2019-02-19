import os
import numpy as np


path = '/home/robbis/mount/permut1/fmri/carlo_ofp/'
#path = "/home/robbis/fmri/carlo_ofp/"
path_remote = "/home/robbis/fmri/data7/Carlo_OFP/"
#path_remote = "/home/robbis/mount/meg_carlo/Carlo_OFP/"
path_remote = "/home/robbis/mount/meg_carlo/data7/Carlo_OFP"

subdir = "analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise/"
subdir = "analysis_SEP/DE_ASS_noHP/"
subdir = "analysis_SEP/DE_ASS_SINGLE_MIXED_EXE/MAGS_MR_UNITS/"


subjects = os.listdir(path_remote)
subjects = [s for s in subjects if s[0] == 's']
subjects.sort()

rm_cmd = ""

for s in subjects:
    orig_dir = os.path.join(s, subdir)
    #command = "cp --parents %s* %s" % (os.path.join(s,subdir), path)
    command = "cp --parents %s*.ifh %s" % (os.path.join(s,subdir), path)
    
    print command
    #os.system(command)
    command = "cp --parents %s*.img %s" % (os.path.join(s,subdir), path)
    print command
    
    command = "cp --parents %s*.txt %s" % (os.path.join(s,subdir), path)
    print command   
    
rm_cmd = ""  
for s in subjects:
    path_subj = os.path.join(path, s, subdir)
    filelist = os.listdir(path_subj)
    filelist = [f for f in filelist if f.find('.ifh') != -1]
    filelist.sort()
    for f in filelist:
        in_file = os.path.join(path_subj, f)
        out_file = os.path.join(path_subj, f[:-9])
        cmd = 'nifti_4dfp -n '+in_file+' '+out_file
        print cmd
        #os.system(cmd)
        rm_cmd += 'rm '+in_file+'\n'
        rm_cmd += 'rm '+in_file[:-4]+'.img\n'
        print cmd
        #os.system(cmd)
    
    rm_cmd += "rm "+os.path.join(path_subj, filelist[0][:-4]+".nii")+"\n"
    
    filelist = filelist[1:] # Remove first regressor movement
    #print filelist
    filetotal = [os.path.join(path_subj, f[:-9]+'.nii') for f in filelist]
    
    cmd = 'fslmerge -t '+os.path.join(path_subj, "residuals.nii.gz")+" "+" ".join(filetotal)
    print cmd
    #os.system(cmd)
    
    conditionlist = [[l for l in f.split('_')[7]] for f in filelist]
    conditionlist = np.array(conditionlist)
    filename = "%s_condition_list.txt" % (s)
    #np.savetxt(os.path.join(path_subj, filename), conditionlist, dtype=np.str)
    #np.savetxt(os.path.join(path_subj, filename), conditionlist, fmt="%s", delimiter=",")
    
print rm_cmd

# copy fidl event to wkpsy01
path_orig = "/home/robbis/mount/meg_carlo/data7/Carlo_OFP/"

path_dest = "/home/robbis/mount/wkpsy01/carlo_ofp/"
path_dest = "/home/robbis/mount/permut1/fmri/carlo_ofp/"

subdir = "analysis_SEP/"

fname = "%s_eventfiles_DE_ASS_SINGLE.txt"
fname = "%s_eventfiles_DE_ASS_MIXED_EXE.txt"


for s in subjects:
    sub_ = s[:4]+s[-6:]
    orig_dir = os.path.join(path_orig, s, subdir, fname % (sub_))
    dest_dir = os.path.join(path_dest, s, subdir)
    
    command = "cp %s %s" % (orig_dir, dest_dir)
    print command
    
  
##
path_orig = "/media/robbis/Seagate_Pt1/data/carlo_ofp/"



for s in subjects:
    sub_ = s[:4]+s[-6:]
    orig_dir = os.path.join(path_orig, s, subdir, fname % (sub_))
    dest_dir = os.path.join(path_dest, s, subdir)
    
    command = "cp %s %s" % (orig_dir, dest_dir)
    print command
  
  
    
### from fidl to pymvpa attributes

path_dest = "/home/robbis/mount/permut1/fmri/carlo_ofp/"
subdir = "analysis_SEP/DE_ASS_noHP/"
fname = "%s_eventfiles_DE_ASS_SINGLE.txt"
from mvpa_itab.utils import beta_attributes, enhance_attributes_ofp, fidl2txt_2
for s in subjects:
    sub_ = s[:4]+s[-6:]
    eventfile = os.path.join(path_dest, s, subdir, fname % (sub_))
    # fidl2txt
    #output_beta = os.path.join(path_dest, s, "%s_eventfile_beta.txt" % (s))
    output_residuals = os.path.join(path_dest, s, "%s_eventfile_residuals.txt" % (s))
    #beta_attributes(eventfile, output_beta, nrun=8)
    #residuals_attributes(eventfile, output_residuals, nrun=8)
    fidl2txt_2(eventfile, output_residuals, runs=8., vol_run=246, stim_tr=6, offset_tr=0)
    enhance_attributes_ofp(output_residuals)
    
    

    

## reordering files
path_dest = "/home/robbis/mount/wkpsy01/carlo_ofp/"
path_dest = "/home/robbis/mount/permut1/fmri/carlo_ofp/"

subdir = "analysis_SEP/DE_ASS_noHP/"
subdir = "analysis_SEP/"

niidir = "analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise/"
niidir = "analysis_SEP/DE_ASS_SINGLE_MIXED_EXE/MAGS_MR_UNITS/"


fname = "%s_eventfiles_DE_ASS_SINGLE.txt"
fname = "%s_eventfiles_DE_ASS_MIXED_EXE.txt"

filepattern = "%s_DE_ASS_noHP_res_SINGLE_%s_mag_333_t88.nii"
filepattern = "%s_DE_ASS_SINGLE_MIXED_EXE_%s_mag_333_t88.nii"
filepattern = "%s_DE_ASS_MIXED_EXE_%s_mag_333_t88.nii"

for s in subjects:
    sub_ = s[:4]+s[-6:]
    eventfname = os.path.join(path_dest, s, subdir, fname % (sub_))
    eventfile = open(eventfname)
    
    eventlist = eventfile.readline().split()
    tr = float(eventlist[0])
    eventlist = eventlist[1:-1]
    
    filelist = [os.path.join(path_dest, s, niidir, filepattern % (sub_, event)) for event in eventlist]
    
    output_fname = os.path.join(path_dest, s, niidir, "residuals_sorted.nii.gz")
    command = "fslmerge -t %s %s" %(output_fname, " ".join(filelist))
    print command
    print "\n\n"

    os.system(command)
    
    
###### MDM Rescue ######
import os
import numpy as np


path_remote = '/home/robbis/mount/meg_carlo/Carlo_MDM/'
path_local = "/media/robbis/DATA/fmri/carlo_mdm/"
subdirs = ["RESIDUALS_MVPA/", "RESIDUALS_MVPA/SINGLE_TRIALS/"]
files_ext = ['nii', 'nii.gz', 'txt']


subjects = os.listdir(path_remote)
subjects = [s for s in subjects if s[0] == '1']
subjects.sort()

rm_cmd = ""

for s in subjects:
    for subdir in subdirs:
        orig_dir = os.path.join(path_remote, s, subdir)
        for ext in files_ext:
            #command = "cp --parents %s* %s" % (os.path.join(s,subdir), path)
            command = "cp --parents %s*.%s %s" % (os.path.join(s, subdir), ext, path_local)
            print(command)
            #os.system(command)

