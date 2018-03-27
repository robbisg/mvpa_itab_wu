from scipy.io import loadmat, savemat
import numpy as np

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
    
############### Conversion #########

path = '/media/robbis/DATA/fmri/movie_viviana/corrected/'
bands = ['alpha', 'beta']
conditions_ = ['rest', 'scramble', 'movie']
mtype = ['masked', 'normal']
key = 'mat_corr_sub_%s%s%s' # condition,

grouped = dict()
min_ = dict()

for band in bands:
    path_ = path+band
    
    for condition in conditions_:
        filelist = os.listdir(path_)
        filelist = [f for f in filelist if f.find(condition) != -1]
        
        for i, f in enumerate(filelist):
            
            data = loadmat(path_+'/'+f)
                        
            run = str(i+1)
            if len(filelist) == 1:
                run = ''
                
            for m in mtype:
                
                mt = '_'+m
                if m == 'normal':
                    mt = ''
                
                d_key = key % (condition.upper(), run, mt)
                dict_key = "%s_%s_%s" % (band, condition, mt)
                
                if i == 0:
                    grouped[dict_key] = []
                    min_[dict_key] = 10000
                
                array_data = data[d_key]
                
                grouped[dict_key].append(array_data)
                min_[dict_key] = array_data.shape[1] if array_data.shape[1] < min_[dict_key] else min_[dict_key]

            del data


for k, array in grouped.iteritems():
    print k
    value = min_[k]
    padded = np.zeros((33, value, 45, 45))
    init = 0
    end = 0
    for m in array:
        
        end = m.shape[0]+init
        padded[init:end,:,:,:] = m[:,:value,:,:]
        init = end
    
    key = k.split('_')
    band = key[0]
    condition = key[1]
    
    mt = key[-1] if key[-1] == 'masked' else 'normal'
    
    out_filename = '/media/robbis/DATA/fmri/movie_viviana/corrected/%s/%s/mat_corr_sub_%s.mat' % (band, mt, condition)
    savemat(out_filename, {'data': padded})
    del padded





