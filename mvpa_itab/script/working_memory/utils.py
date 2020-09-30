###################
import numpy as np
from scipy.io import loadmat, savemat
import os
import glob
import warnings
warnings.filterwarnings("ignore")

################################################

# Correction of connectivity data
path = "/media/robbis/DATA/fmri/working_memory/"
path = "/media/robbis/Seagate_Pt1/data/working_memory/"
subject_list = glob.glob(path+"data/sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "meg_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "connectivity_matrix.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]

        if i < 57:
            size_mat = loadmat(os.path.join(path, "parcelsizes_%s.mat" % (band)))
            idx = i
        else:
            size_mat = loadmat(os.path.join(path, "parcelsizes_%s_NEW.mat" % (band)))
            idx = i - 57
        
        size_key = "%s%s" % (band, condition)
        size = np.expand_dims(size_mat[size_key][idx], axis=0)
        
        #size = np.expand_dims(size_mat[size_key].mean(0), axis=0)

        size = np.int_(size)
        size_matrix = np.dot(size.T, size)

        norm_matrix = matrix / np.float_(size_matrix)
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "mpsi_normalized.mat"), norm_ds)


####################################################
# Correction based on rest #

path = "/media/robbis/DATA/fmri/working_memory/"
subject_list = glob.glob(path+"sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "meg_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "mpsi_normalized.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    baseline_rest = dict()

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]
        if condition == 'rest':
            baseline_rest[band] = matrix


    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]

        norm_matrix = matrix / baseline_rest[band]
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "mpsi_rest_normalized.mat"), norm_ds)


####################################################
# Correction of power #

path = "/media/robbis/DATA/fmri/working_memory/"
subject_list = glob.glob(path+"data/sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "power_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "power_parcel.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]


        if i < 57:
            size_mat = loadmat(os.path.join(path, "parcelsizes_%s.mat" % (band)))
            idx = i
        else:
            size_mat = loadmat(os.path.join(path, "parcelsizes_%s_NEW.mat" % (band)))
            idx = i - 57        
        
        size_key = "%s%s" % (band, condition)

        #size = np.int_(size_mat[size_key].mean(0))

        size = size_mat[size_key][idx]
        norm_matrix = data[j] * size
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "power_normalized.mat"), norm_ds)

# Copying script
subject_list = glob.glob(path+"data/sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    folder = subj.split("/")[-1]
    command = "cp %s %s" % (os.path.join(subj, "meg", "power_normalized.mat"), 
                            "/home/robbis/mount/triton.aalto.fi/data/"+folder+"/meg/")
    print(command)
    command = "cp %s %s" % (os.path.join(subj, "meg", "mpsi_normalized.mat"), 
                            "/home/robbis/mount/triton.aalto.fi/data/"+folder+"/meg/")
    print(command)




