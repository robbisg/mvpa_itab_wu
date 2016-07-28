from scipy.io import loadmat, savemat

filename = '/home/robbis/Share/viviana-2016/all_mat_nonstat_beta.mat'
mat_ = loadmat(filename)

keys_ = mat_.keys()

keys_ = [k for k in keys_ if k.find('mat_') != -1]

for k in keys_:
    out_filename = '/media/robbis/DATA/fmri/movie_viviana/%s' % (k)
    savemat(out_filename, {'data': mat_[k]})

