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
path = '/media/guidotr1/Seagate_Pt1/data/Viviana2018/'
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
                dict_key = "%s_%s_%s" % (band, condition, m)
                
                if i == 0:
                    grouped[dict_key] = []
                    min_[dict_key] = 10000
                
                array_data = data[d_key]
                
                grouped[dict_key].append(array_data)
                min_[dict_key] = array_data.shape[1] if array_data.shape[1] < min_[dict_key] else min_[dict_key]

            del data
            gc.collect()

minimum = np.min(list(min_.values()))


for k, array in grouped.iteritems():
    print(k)
    darray = np.float16(np.vstack([a[:, :minimum, :, :] for a in array]))
    grouped[k] = darray
    

subj_dict = dict()

triu = np.triu_indices(45, k=1)

for i in range(33):

    key = "subj_%02d" % (str(i+1))
    data = np.float16(np.array([array[i, :, triu[0], triu[1]] for k, array in grouped.items()]))
    os.makedirs(os.path.join(path, "meg", key, "meg"))
    fname = os.path.join(path, "meg", key, "meg", "connectivity_matrix.mat")
    savemat(fname, {'data': data})
    del data
    gc.collect()




    for k, result in results.items():
        path_ = os.path.join(path, subject, 'meg')
        command = "mkdir -p "+path_
        os.system(command)
        attributes = [['targets','band','run']]
        data = []
        for label, matrix in result.items():
            band, condition, run = label.split("_")
            data.append(matrix)
            attributes.append([condition, band, run]) 
            
        



attributes = [['band','target','mtype']]
attributes += [k.split("_") for k, v in grouped.items()]
np.savetxt(os.path.join(path, "attributes.csv"), 
                   np.array(attributes, dtype=np.str_), 
                   fmt="%s", delimiter=",", 
                   #header=['condition', 'band', 'run']
                   )


