import glob
import numpy as np
import os
from scipy.io import savemat

path = "/media/robbis/DATA/fmri/monks/"
directory = glob.glob(path+"0_results/*connectivity*")

subjects = os.listdir(directory[0])
subjects = [s for s in subjects if s[0].isnumeric()]
subjects.sort()

for dir_ in directory:
    for subj in subjects:

        matrices_txt = os.listdir("%s/%s/" % (dir_, subj))
        matrices_txt.sort()

        attributes = [['targets', 'run']]
        data = []

        for m in matrices_txt:
            _, condition, _, run = m[:-4].split("_")

            attributes.append([condition, run])
            matrix = np.loadtxt('%s/%s/%s' % (dir_, subj, m))
            matrix[np.isnan(matrix)] = 0.
            matrix[np.isinf(matrix)] = 0.
            data.append(matrix)

        folder = dir_.split("/")[-1]
        fname = os.path.join(path, subj, "fcmri", folder+".mat")
        savemat(fname, {"data":np.array(data)})

        fname = os.path.join(path, subj, "fcmri", "attributes_"+folder+".txt")
        np.savetxt(fname, 
                   np.array(attributes, dtype=np.str_), 
                   fmt="%s", delimiter=" ", 
                   #header=['condition', 'band', 'run']
                   )
        




