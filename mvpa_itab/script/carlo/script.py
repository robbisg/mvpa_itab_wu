import os
import numpy as np


path = '/root/robbis/DATA/fmri/carlo_ofp/'
#path = "/home/robbis/fmri/carlo_ofp/"
path_remote = "/home/robbis/fmri/data7/Carlo_OFP/"
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
    