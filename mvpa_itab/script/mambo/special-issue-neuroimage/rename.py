import h5py
import hdf5storage
import numpy as np
from scipy.io import loadmat, savemat

shared = "/run/user/1000/gvfs/smb-share:server=192.168.30.54,share=meg_data_analisi/"
path = os.path.join(shared, "HCP_Motor_Task_analysis")
subjects = os.listdir(path)
subjects.sort()

bids_path = "/media/robbis/HP x755w/hcp_motor/"

label_dict = {1: "LH", 2: "LF", 4: "RH", 5: "RF", 6: "FIX"}

for subj in subjects:
    subj_path = os.path.join(path, subj)
    files = os.listdir(subj_path)
    files.sort()

    bids_subj = os.path.join(bids_path, "sub-%s" % (subj), 'meg')
    os.system("mkdir -p %s" % (bids_subj))

    for s, f in enumerate(files):
        fname = os.path.join(subj_path, f)
        mat = h5py.File(fname)

        for k in ['powerbox', 'trailinfo']:
            
            if k == 'powerbox':
                fsave = "sub-%s_ses-%02d_task-motor_kind-powerbox_meg.mat" % (subj, s)
                #savemat(os.path.join(bids_subj, fsave), {'data': data}, do_compression=True)
                #hdf5storage.savemat(os.path.join(bids_subj, fsave), {'data': data}, store_python_metadata=False)


            if k == 'trailinfo':
                data = np.squeeze(mat[k][:])
                header = ['vec1', 'labels', 'chunks', 'vec2', 'vec3', 'rt', 'vec4',
                          'targets', 'side', 'part']
                data = data.T
                targets = np.array([[label_dict[l] for l in data[:,1]]])
                side = np.array([[t[0] for t in targets[0]]])
                part = np.array([[t[1] for t in targets[0]]])

                data = np.hstack((data, targets.T, side.T, part.T))
                fsave = "sub-%s_ses-%02d_task-motor_kind-powerbox_events.tsv" % (subj, s)
                events_fname = os.path.join(bids_subj, fsave)
                np.savetxt(events_fname, data, fmt="%s", delimiter="\t", header="\t".join(header))


###################
