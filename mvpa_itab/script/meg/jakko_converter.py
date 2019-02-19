from scipy.io import loadmat
import os
import numpy as np
from mvpa_itab.conn.operations import copy_matrix
from scipy.io.matlab.mio import savemat



def get_full_matrix(matrix):
    
    nzx = np.nonzero(matrix[0])
    nzy = np.nonzero(matrix[:,-1])
    
    nzm = np.nonzero(matrix)
    
    grid = np.meshgrid(nzx, nzy)
        
    full_matrix = np.zeros((matrix.shape[1], matrix.shape[1]))
    
    grid_matrix = full_matrix[grid]
    idx = np.triu_indices(grid_matrix.shape[0], k=0)
    
    grid_matrix[idx] = matrix[nzm]
    full_matrix[grid] = grid_matrix
    
    full_matrix = copy_matrix(full_matrix.T, diagonal_filler=0.)
    
    return full_matrix


def get_full_power(x):
    data = loadmat("/media/robbis/DATA/fmri/working_memory/sub_01/meg/mpsi_normalized.mat")
    matrix = data['data']
    parcels = matrix[0].shape[0]
    idx = np.nonzero((matrix[0] != 0).sum(0))[0]

    full_array = np.zeros(parcels)

    power = np.vstack([s.squeeze() for s in x.squeeze()]).squeeze()

    full_array[idx] = power

    return full_array


path = "/media/robbis/DATA/fmri/working_memory/"
n_subjects = 57

def get_subject_results(path, pattern="MPSI", n_subjects=57, read_fx=get_full_matrix):

    if pattern=="MPSI":
        len_cond = len(pattern)
    else:
        len_cond = len("POWER")
        
    #path = "/media/robbis/DATA/fmri/working_memory/"
    list_mat = os.listdir(path)
    list_mat = [m for m in list_mat if m.find(pattern) != -1]
    results = dict()
    
    #n_subjects = 57
    
    for m in list_mat:
        band = m[len(pattern):-4]
        #band = m[4:-4]
        mat_file = loadmat(os.path.join(path, m))
        conditions = [k for k in mat_file.keys() if k[0] != '_']
        
        for c in conditions:
            data = mat_file[c]
            
            for s in range(data.shape[0]):
                session = data[s]
                
                for subject in range(len(session)):
                    matrix = session[subject]
                    
                    full_matrix = read_fx(matrix)
                    key = "sub_%02d" %(subject+1)
                    
                    if not key in results.keys():
                        results[key] = dict()
                    
                    key_cond = "%s_%s_%s" % (band, c[len_cond:], str(s+1))
                    results[key][key_cond] = full_matrix
    return results
                
                
# store results

def store_results(path, results, fname="connectivity_matrix.mat", attr="meg_attributes.txt"):

    fname = "power_parcel.mat"
    attr  = "power_attributes.txt"
    
    for subject, result in results.items():
        path_ = os.path.join(path, subject, 'meg')
        command = "mkdir -p "+path_
        os.system(command)
        attributes = [['targets','band','run']]
        data = []
        for label, matrix in result.items():
            band, condition, run = label.split("_")
            data.append(matrix)
            attributes.append([condition, band, run]) 
            
        
        np.savetxt(os.path.join(path, subject, attr), 
                   np.array(attributes, dtype=np.str_), 
                   fmt="%s", delimiter=" ", 
                   #header=['condition', 'band', 'run']
                   )
        
        savemat(os.path.join(path_, fname), {'data':np.array(data)})
    
    

subjects = results.keys()
subjects.sort()

subjects = [[s, i+1] for i, s in enumerate(subjects)]

    
    
path = "/media/robbis/DATA/fmri/working_memory/"
list_mat = os.listdir(path)
list_mat = [m for m in list_mat if m.find("MPSI") == -1 and m.find(".mat") != -1]

vector = []
header = ["subjects", "idx"]
for b in list_mat:
    mat = loadmat(os.path.join(path, b))
    for k in mat.keys():
        if k[0] == '_':
            continue
        header.append(k+'_1')
        header.append(k+'_2')
        data = mat[k].squeeze()
        vector.append(data[::2])
        vector.append(data[1::2])

vector = np.array(vector).T
np.savetxt(os.path.join(path, "participants.csv"), 
           np.array(np.hstack((subjects, vector)), dtype="|S6"), 
           fmt="%s", delimiter=",", comments="",
           header=",".join(header)
           )        
              
    
