from scipy.io import loadmat, savemat

filename = '/home/robbis/Share/viviana-2016/all_mat_nonstat_beta.mat'
data = loadmat(filename)

keys_ = data.keys()

keys_ = [k for k in keys_ if k.find('mat_') != -1 and k.find("masked") != -1]
keys_.sort()
conditions_ = ['MOVIE', 'SCRAMBLE', 'REST']

grouped = dict()
max_ = dict()
for c in conditions_:
    grouped[c] = []
    max_[c] = 0
    for k in keys_:
        if k.find(c) != -1:
            grouped[c].append(data[k])
            if data[k].shape[1] > max_[c]:
                max_[c] = data[k].shape[1]
            


for k, array in grouped.iteritems():
    print k
    value = max_[c]
    padded = np.zeros((33, value, 45, 45))
    init = 0
    end = 0
    for m in array:
        print init, end
        end = m.shape[0]+init
        padded[init:end,:m.shape[1],:,:] = m
        init = end
        
    out_filename = '/media/robbis/DATA/fmri/movie_viviana/alpha/masked/mat_corr_sub_%s.mat' % (k)
    savemat(out_filename, {'data': padded})