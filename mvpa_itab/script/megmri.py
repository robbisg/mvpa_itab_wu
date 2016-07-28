from PIL import Image
from scipy.stats import kendalltau
from sklearn.metrics import jaccard_similarity_score, normalized_mutual_info_score
import itertools
import os
import numpy as np
from nipy.algorithms.registration.resample import resample

import inspect

import sklearn
import scipy
sklearn_functions = inspect.getmembers(sklearn.metrics, inspect.isfunction)
distance = inspect.getmembers(scipy.spatial.distance, inspect.isfunction)
 
distance = [d for d in distance if d[0][0] != '_']


def filter_image(image, i, j, window, func=np.mean):
    x = i*window
    y = j*window
    return func(image[x:x+window, y:y+window])


def resample(image, window_size):
    
    new_dim = np.rint(np.array(image.shape)/np.float(window_size))
    img_new = np.zeros(np.int_(new_dim))
    
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            img_new[i,j] = filter_image(image, i, j, window_size)
            
    return img_new


path_dict = {'SPM': '/home/robbis/phantoms/Coregistration/Comparison/SPM/',
             'FSL': '/home/robbis/phantoms/Coregistration/Comparison/FSL/',
             'BV' : '/home/robbis/phantoms/Coregistration/Comparison/BV/',
             'SlidingWindow': '/home/robbis/phantoms/Coregistration/Comparison/SlidingWindow/',
             'LF': '/home/robbis/phantoms/Coregistration/Comparison/LF116/'}
# Load images
images_ = dict()
for k, path in path_dict.iteritems():
    file_list = os.listdir(path)
    file_list.sort()
    images_[k] = [np.array(Image.open(os.path.join(path, f))) for f in file_list]
    images_[k] = np.array(images_[k])


all_functions = distance+sklearn_functions+[('kendalltau', kendalltau)]

dict_functions = dict(all_functions)
    
for k, image_stack in images_.iteritems():
    im_stack = []
    for i, image in enumerate(image_stack):
        # Downsample if necessary
        
        if k in ['SPM', 'FSL']:
            print k
            rimage = resample(image, 4)
            rimage = np.pad(rimage, ((9, 9), (9,9)), mode='minimum')
            image = rimage.copy()
            im_stack.append(image)
    
    if k in ['SPM', 'FSL']:
        images_[k] = np.array(im_stack)
        
        



target = ['LF']
metrics_ = []
for name, fx in dict_functions.iteritems():
    stack_ = []
    for k, image_stack in images_.iteritems():
        if k not in target:
            try:
                m = fx(np.int_(image_stack[:6,...].flatten()), images_['LF'].flatten())
            except Exception, err:
                print name, err
                continue
            stack_.append((k, m))
        
    metrics_.append((name, dict(stack_)))

metrics_ = dict(metrics_)
keys_ = metrics_.keys()

droppable_keys = ['precision_recall_fscore_support', 
                  'pairwise_distances_argmin', 
                  'is_valid_dm', 
                  'homogeneity_completeness_v_measure', 
                  'classification_report', 
                  'adjusted_mutual_info_score',
                  '_nbool_correspond_ft_tf',
                  'confusion_matrix'
                  ]

filtered_metrics = []
for k, m in metrics_.iteritems():
    if k not in droppable_keys and m != {}:
        filtered_metrics.append((k, m))

metrics_ = dict(filtered_metrics)



def build_path_dict(path, **kwargs):
    
    levels = []
    for k, v in kwargs.iteritems():
        levels.append(v)
        
    
    comb_ = list(itertools.product(*levels))
    
    possibilities_ = dict()
    for comb in comb_:
        key_ = '_'.join(comb)
        value_ = os.path.join(path, '/'.join(comb))
        possibilities_[key_] = value_
        
    return possibilities_
        
list_metrics = []
for m, i in zip(selected_keys, selected_metrics):
    list_ = [[],[],[]]
    for k, v in i.iteritems():
        for j, d in enumerate(dirs_):
            if k.find(d) != -1:
                if m=='kendalltau':
                    value = v.correlation
                else:
                    value = v
                list_[j].append(value)
    list_metrics.append(list_)


x = np.arange(6)
selected_keys = ['kendalltau',
                 'correlation',
                 'braycurtis',
                 'jaccard',
                 'mean_squared_error',
                 'r2_score']
labels = ['rigid', 'zoom', 'shear']
markers = ['o', 's', '^']

for i, metric in enumerate(selected_keys):
    pl.figure()
    series = metrics_sel[i]
    for j, y in enumerate(series):
        pl.plot(x, y, lw=2.5, alpha=0.5, marker=markers[j], label=labels[j])
    pl.title(metric)
    pl.legend()
    

mutual = np.array([[0.916632,0.915068,0.916718],
                   [0.916551,0.915684,0.917098],
                   [0.916832,0.915901,0.916693],
                   [0.917053,0.915852,0.917052],
                   [0.916621,0.916179,0.916621],
                   [0.916604,0.91509,0.916316]]).T

############### converti file tif ################
path_ = '/home/robbis/phantoms/TIFF/PHANTOM_2015/HF_N/'
lista_file = os.listdir(path_)

for f_ in lista_file:
    
    input_ = os.path.join(path_, f_)
    output_ = os.path.join(path_, 'strip_'+f_)
    print 'tiffcp -r 1 -c none %s %s' % (input_, output_) 
  
  
################# convert mat phantoms in tif ##############
from scipy.io import loadmat
from PIL import Image

path = '/home/robbis/phantoms/TIFF/PHANTOM_2015/mat/'
path = '/home/robbis/phantoms/TIFF/PHANTOM_2015/'
lista_file = os.listdir(path)
lista_file = [f for f in lista_file if f.find('.mat') != -1]

for f in lista_file:
    
    mat_ = loadmat(os.path.join(path, f))
    #stack_ = mat_['pippopippo']
    stack_ = mat_['image']
    #print stack_.max()
    stack_ = np.abs(stack_)
    stack_ = np.array(255*stack_/stack_.max(), np.uint8)
    
    dir_ = f[11:-4]
    command_ = 'mkdir '+os.path.join(path, dir_)
    os.system(command_)
    
    for j in range(stack_.shape[1]):
        img_ = stack_[:,j,:]
        fname_ = '%s_%02d.tif' %(f[:-4], j)
        #tif_ = np.array(256*img_, dtype=np.uint8)
        im = Image.fromarray(img_)
        input_ = os.path.join(path, dir_, fname_)
        im.save(input_)
        output_ = os.path.join(path, dir_, '%s_cp_%02d.tif' %(f[:-4], j))
        convert_ = 'tiffcp -r 1 -c none %s %s' % (input_, output_)
        os.system(convert_)
        
        rm_ = 'rm %s' %(input_)
        os.system(rm_)
        
####################################################
        
lista_file = os.listdir(path)
path_2 = '/home/robbis/development/svn/mutual/trunk/fileList/'
lista_file = [f for f in lista_file if f.find('mat') == -1]
for f in lista_file:
    dir_ = os.path.join(path, f, '*')
    file_ = os.path.join(path_2, 'listaLF_phantom_'+f+'.txt')
    command_ = 'ls %s > %s' %(dir_, file_)
    print command_
  
  
####################################
path = '/home/robbis/development/svn/mutual/trunk/fileList/'
lista_file = os.listdir(path)

lista_file = [f for f in lista_file if f.find('listaLF_phantom') != -1 and f.find('txt~') == -1]
for f in lista_file:
    
    input_fname = os.path.join(path, f)
    output_fname = os.path.join('/home/robbis/development/svn/mutual/trunk/result',f[:-4]+'_best.txt')
    command = './Mutual.x %s %s > %s' % (input_fname, str(9), output_fname)
    print command
    os.system(command)
    
##########################################
file_name = '/home/robbis/development/svn/mutual/trunk/fileList/listaLF_phantom_37ave.txt'

for i in [6, 9, 12]:
    out_name = '/home/robbis/development/svn/mutual/trunk/result/37ave_%s.txt' % (str(i))
    command = './Mutual.x %s %s > %s' % (file_name, str(i), out_name)
    print command
    os.system(command)
  
