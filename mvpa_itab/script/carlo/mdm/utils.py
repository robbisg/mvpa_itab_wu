import pandas as pd
from pyitab.analysis.results import filter_dataframe
import numpy as np
import os


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

event_file = pd.read_excel("/home/robbis/Downloads/group_4MVPA (1).xls")
event_file['image_cat_num'] = le.fit_transform(df_num['image_cat'].values)


subjects = np.unique(event_file['name'].values)

path_wk = "/home/robbis/mount/meg_workstation/Carlo_MDM/%s/RESIDUALS_MVPA/beta_attributes_full.txt"
folders = os.listdir("/home/robbis/mount/meg_workstation/Carlo_MDM/")

name_dict = {}

for subj in subjects:

    subdir = folders[np.nonzero([s.find(subj) != -1 for s in folders])[0][0]]

    name_dict[subj] = subdir

    df = filter_dataframe(event_file, name=[subj])
    
    df.to_csv(path_wk % (subdir), index=False, sep=" ")



name_dict = {}

for subj in subjects:

    subdir = folders[np.nonzero([s.find(subj) != -1 for s in folders])[0][0]]

    name_dict[subj] = subdir

event_file['name'] = [name_dict[name] for name in event_file['name'].values]



df_num = event_file.copy()

for field in df_num.keys():
    values = le.fit_transform(df_num[field].values)
    #values -= values.mean()
    df_num[field] = values


######################## REST ##########################
path_rest = "%s/boldrest_%d/"

subjdirs = glob.glob("/home/robbis/mount/meg_workstation/Carlo_MDM/LOCALIZER/*")

subject_list = []
no_rest = []
for s in subjects:
    rest_files = []
    dir_ = [d for d in subjdirs if d.find(s) != -1]
    if len(dir_) == 0:
        no_rest.append(s)
        continue
    else:
        for d in dir_:
            rest_files += [glob.glob(path_rest % (d, i+1))+"*bpss_resid*") for i in range(3)]
    
    subject_list.append(rest_files)


flat_subject_list = [i for item in subject_list for i in item]
flat_subject_list = [i for item in flat_subject_list for i in item]


path_annalisa = "/home/robbis/mount/meg_workstation/Annalisa_DecisionValue/"
subdirs =  glob.glob(path_annalisa+"*")

subject_list = []
for s in no_rest:
    dir_ = [d for d in subdirs if d.find(s) != -1]
    dir_.sort()
    if len(dir_) == 0:
        #no_rest.append(s)
        continue
    else:
        dir_ = dir_[1]
    rest_files = [glob.glob(os.path.join(dir_,"boldrest_%s"%(i+1), "*bpss_resid*")) for i in range(3)]
    subject_list.append(rest_files)

flat_subject_list2 = [i for item in subject_list for i in item]
flat_subject_list2 = [i for item in flat_subject_list2 for i in item]


lista_files = [l for l in lista_files if l.find('.ifh') != -1]

path_mdm = "/home/robbis/mount/meg_workstation/Carlo_MDM/"
path_permut1 = "/home/robbis/mount/permut1/fmri/carlo_mdm/"
subjdirs =  os.listdir(path_permut1)
subjdirs = [s for s in subjdirs if s[:2].isnumeric()]
for dir_ in subjdirs:
    subj = dir_[6:]


    #files_ = [f for f in lista_files if f.find(subj) != -1]

    rest_dir = os.path.join(path_permut1, dir_, "rest")



    bold_dir = os.listdir(os.path.join(path_mdm, dir_))

    bold_dir = [b for b in bold_dir if b.find('bold') != -1]

    attributes = [['targets', 'chunks']]

    for i, f in enumerate(range(3)):
        #fname = f.split('/')[-1]
        #nii = os.path.join(rest_dir, fname[:-4]+'.nii.gz.nii')
        #command = "rm %s" % (nii)
        #print(command)
        
        #nii = os.path.join(rest_dir, fname[:-4])
        #print("rm %s.nii" % (nii))
        command = "nifti_4dfp -n %s %s" % (f, os.path.join(rest_dir,"rest_rest_%02d" % (i+1)))
        try:
            img = ni.load(os.path.join(rest_dir,"rest_rest_%02d.nii" % (i+1)))
        except Exception:
            continue
        volumes = img.shape[-1]
        for _ in range(volumes):
            attributes.append(['rest', '%s' % (i+1)])

        #print("%s %s" % (nii, img.shape))
        #print(command)



    for i in range(len(bold_dir)):
        files_task = glob.glob(os.path.join(path_mdm, dir_, 'bold%d'%(i+1), '*bpss_resid*.ifh'))
        fname_out = os.path.join(rest_dir, 'task_rest_%02d' % (i+1))
        
        command = "nifti_4dfp -n %s %s" % (files_task[0], fname_out)
        #os.system(command)

        img = ni.load(fname_out+".nii")
        volumes = img.shape[-1]
        for _ in range(volumes):
            attributes.append(['task', '%s' % (i+1)])
        #print("%s %s" % (fname_out, img.shape))
        
    fname = "attributes_rest.csv"
    np.savetxt(os.path.join(rest_dir, fname), 
               np.array(attributes, dtype=np.str), 
               fmt="%s",
               delimiter=' ')







################# AFNI MASKS ######################
import glob
import os
path = '/media/robbis/DATA/fmri/carlo_mdm/0_results/derivatives/'

for tmap in glob.glob(path+"*.HEAD"):
    task = os.path.basename(tmap).split('_')[0]
    mask_fname = os.path.join(path, task)
    command = "3dclust -1Dformat -nosum -1dindex 0 -1tindex 1 -2thresh -2.807 2.807 -dxyz=1 -savemask %s_mask 1.01 20 %s"
    command = command % (mask_fname, tmap)
    print(command)
    
    os.system(command)
    output = "%s%s_conjunction" % ( path, task)
    command = "3dcalc -a %s_mask+tlrc -b %s[1] -expr '(a/a)*b' -prefix %s" %(mask_fname, tmap, output)
    print(command)
    os.system(command)
    
    output_fname = "%s.nii.gz" % (output)
    command = "3dTcat -prefix %s %s+tlrc[0]" % (output_fname, output)
    print(command)
    os.system(command)
    
    img = ni.load(output_fname)
    output = img.get_data().squeeze()
    
    save_map(output_fname, output, affine=img.affine)


path = '/media/robbis/DATA/fmri/carlo_mdm/0_results/derivatives/'

for tmap in glob.glob(path+"*mask+tlrc.HEAD"):
    task = os.path.basename(tmap).split('_')[0]
    mask_fname = os.path.join(path, task)

    output_fname = "%s.nii.gz" % (tmap[:-10])
    command = "3dTcat -prefix %s %s" % (output_fname, tmap[:-5])

    print(command)


###########################à

big_table = {}
path = '/media/robbis/DATA/fmri/carlo_mdm/1_single_ROIs/'

for fname in glob.glob(path+"*mask.nii.gz"):
    print(fname)
    img = ni.load(fname)
    center_711 = np.array([-70.5, -105, -60.])
    data = img.get_data().squeeze()
    table = []
    for f in np.unique(data)[1:]:
        mask_roi = data == f

        center_mass = np.mean(np.nonzero(mask_roi), axis=1)
        x,y,z = np.rint([3,3,3]*center_mass+center_711)

        command = "whereami %s %s %s -lpi -space TLRC -tab" %(str(x+2.), str(y+2), str(z+2))
        var = os.popen(command).read()
        lines = var.split("\n")
        index = [i for i, l in enumerate(lines) if l[:5] == 'Atlas']
        label1 = lines[index[0]+1]
        label2 = lines[index[0]+2]
        if label1[0] == '*':
            area1 = area2 = "None"
        else:
            area1 = label1.split("\t")[2]
            area2 = label2.split("\t")[2]
        table.append([x, y, z, area1, np.count_nonzero(mask_roi), f])
        print(center_mass, x, y, z, area1, f)
    key = fname.split('/')[-1].split('.')[0][:-5]
    big_table[key] = table






path_wk = "/media/robbis/DATA/fmri/carlo_mdm/"
path_meg = "/home/robbis/mount/meg_workstation/Carlo_MDM/"
path_meg = "/home/robbis/mount/permut1/fmri/carlo_mdm/"
pattern = "%s/RESIDUALS_MVPA/beta_attributes_full.txt"
subjects = os.listdir("/media/robbis/DATA/fmri/carlo_mdm/")

subjects = [s for s in subjects if s[:2].isnumeric()]

for subj in subjects:

    orig = os.path.join(path_meg, pattern % (subj))
    dest = os.path.join(path_wk, pattern % (subj))

    command = "cp %s %s" % (dest, orig)
    print(command)
    os.system(command)



# Residuals full attribute
path_wk = "/media/robbis/DATA/fmri/carlo_mdm/%s/RESIDUALS_MVPA/residuals_attributes_full.txt"
subjects = os.listdir("/media/robbis/DATA/fmri/carlo_mdm/")

subjects = [s for s in subjects if s[:2].isnumeric()]

fix_columns = ['image_type', 'decision', 'evidence', 'accuracy',
       'memory_status', 'motor_resp', 'target_side', 'image_cat',
       'image_cat_num']

for subj in subjects:
    df_residuals = pd.read_csv("/media/robbis/DATA/fmri/carlo_mdm/%s/RESIDUALS_MVPA/attributes_residuals_plus.txt" % (subj), 
                            delimiter=" ")
    drop_keys_residuals = ['evidence', 'accuracy', 'memory_status', 'decision']
    df_residuals = df_residuals.drop(drop_keys_residuals, axis=1)

    df_beta = pd.read_csv("/media/robbis/DATA/fmri/carlo_mdm/%s/RESIDUALS_MVPA/beta_attributes_full.txt"% (subj), 
                        delimiter=" ")
    drop_keys_beta = ['chunks', 'targets', 'trial', 'block']
    df_beta = df_beta.drop(drop_keys_beta, axis=1)

    trials, count = np.unique(df_residuals['trial'].values, return_counts=True)

    df_beta_full = pd.DataFrame()
    for trial in trials:
        df_partial = pd.concat([df_beta.loc[trial:trial] for _ in range(count[trial])], ignore_index=True, axis=0)
        df_beta_full = pd.concat([df_beta_full, df_partial], ignore_index=True)

    df_residuals_full = pd.concat([df_residuals, df_beta_full], axis=1)
    mask = np.logical_and(df_residuals_full['targets'] == 'FIX', 
                          df_residuals_full['frame'] != 1)
    for col in fix_columns:
        label = 'F'
        if col in ['evidence', 'image_cat_num']:
            label = 0
        df_residuals_full.loc[mask, col] = label

    df_residuals_full.to_csv(path_wk % (subj), index=False, sep=" ")