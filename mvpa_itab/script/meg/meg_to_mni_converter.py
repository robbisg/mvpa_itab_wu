import h5py
from scipy.io.matlab.mio import loadmat
import nibabel as ni



def mat_to_mni(flatten_map, voxel_position):
    mask = ni.load("/media/robbis/DATA/fmri/templates_MNI_3mm/mask_MNI_4mm_meg.img")
    seed = np.zeros_like(mask.get_data(), dtype=np.float)
    
    for i, pos in enumerate(voxel_position.T):
        x, y, z = pos
        
        seed[x,y,z] = flatten_map[i]
        
    seed = ni.Nifti1Image(seed, mask.affine)
    return seed


path = "/media/robbis/DATA/fmri/monks"
path_meg = "/media/robbis/DATA/fmri/monks/meg/"

filelist = os.listdir(path_meg)
filelist = [f for f in filelist if f.find("new") != -1]

subjects = ['061110jutpre',
            '061107phimoo',
            '061109sawsam',
            '061102jakbar',
            '061103kuatsa',          
            '061102chrwoo',
            '061107lucpri',
            '100607intpra',
            '090709phrjan',
            ]


voxel_position = loadmat(os.path.join(path_meg, 'voxel_position.mat'))
voxel_position = voxel_position['voxel_position']

conditions = {'VIP': 'vipassana', 'SAM':'samatha', 'REST':'rest'}

dataset = {s: [] for s in subjects}


for f in filelist:
    
    mat = h5py.File(os.path.join(path_meg, f))
    condition = conditions[f.split("_")[1]]
    seed = f.split("_")[-2]
    
    for key in mat.keys():
        
        rithm = key.split("_")[-1]
        
        data = mat[key].value
        
        for i, map in enumerate(data.T):
            mni_map = mat_to_mni(map, voxel_position)
            
            dataset[subjects[i]].append([seed, rithm, condition, condition, mni_map])
            

            

attributes = []
header = "seed band condition targets chunks"
for s in dataset.keys():
    full_data = dataset[s]
    attributes = [c[:4]+['0'] for c in full_data]
    data = np.array([c[4].get_data() for c in full_data])
    
    data = np.rollaxis(data, 0, 4)
    attributes = np.array(attributes, dtype="|S10")
    
    dir_ = os.path.join(path, s, 'meg')
    command = "mkdir -p %s" % (dir_)
    print command
    os.system(command)
    
    fname = os.path.join(dir_, "seed_map.nii.gz")
    ni.save(ni.Nifti1Image(data, c[4].affine), fname)
    
    fname = os.path.join(path, s, 'attributes_meg.txt')
    np.savetxt(fname, attributes, fmt='%s', delimiter=' ', header=header, comments='')
            
        
        
        
        
        
        