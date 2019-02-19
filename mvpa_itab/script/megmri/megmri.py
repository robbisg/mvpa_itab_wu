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
            print(k)
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
            except Exception as err:
                print(name, err)
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

selected_keys = ['kendalltau',
                 'correlation',
                 'braycurtis',
                 'jaccard',
                 'mean_squared_error',
                 'r2_score']
values_ = [metrics_[k] for k in selected_keys]

def build_path_dict(path, **kwargs):
    """
    Input like
    kwargs = {'level1':['3_trial_sheer', '2_trial_zoom', '1_trial'], 
              'level2':['offset_0_param_9', 
                        'offset_1_param_9',
                        'offset_2_param_9',
                        'offset_3_param_9',
                        'offset_4_param_9',
                        'offset_5_param_9']}
    path = '/home/robbis/development/svn/mutual/trunk/result/MARZO16/'
    
    returns:
        a dictionary with all possibilities between level1 dirs and level2 dirs
    """     
    
    
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


path_dict = build_path_dict(path, **kwargs)
path_dict.update({'LF': '/home/robbis/phantoms/Coregistration/Comparison/LF116/'})

images_ = dict()
for k, path in path_dict.iteritems():
        file_list = os.listdir(path)
        file_list.sort()
        images_[k] = [np.array(Image.open(os.path.join(path, f))) for f in file_list]
        images_[k] = np.array(images_[k])


selected_keys = ['kendalltau',
                 'correlation',
                 'braycurtis',
                 'jaccard',
                 'mean_squared_error',
                 'r2_score']


selected_metrics = [metrics_[k] for k in selected_keys]
dirs_ = ['1_trial','2_trial', '3_trial']
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
    
mutual = np.array([[0.916632,0.915068,0.916718],
                   [0.916551,0.915684,0.917098],
                   [0.916832,0.915901,0.916693],
                   [0.917053,0.915852,0.917052],
                   [0.916621,0.916179,0.916621],
                   [0.916604,0.91509,0.916316]]).T

list_metrics.append(1/(mutual-0.0058))

x = np.arange(6)
selected_keys = ['kendalltau',
                 'correlation',
                 'braycurtis',
                 'jaccard_similarity_score',
                 'mean_squared_error',
                 'r2_score',
                 'nmi']

graph_title = ['Kendall Tau',
               'Correlation distance',
               'Bray-Curtis distance',
               'Jaccard correlation',
               'Mean Squared Error',
               'R2 score',
               'Normalized Mutual Information']

labels = ['rigid', 'zoom', 'shear']
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']


metrics_sel = np.array(list_metrics)
fig = pl.figure(figsize=(23,12), dpi=150)

for i, metric in enumerate(selected_keys):
    ax = fig.add_subplot(2,4,i+1)
    series = metrics_sel[i]
    lines = []
    for j, y in enumerate(series):
        l = ax.plot(x, y, lw=2.5, alpha=0.8, markersize=12, marker=markers[j], label=labels[j], color=colors[j])
        ax.hlines(y.mean(), x[0], x[-1], alpha=0.7, linestyles='dashed', color=colors[j])
        lines.append(l[0])
    ax.set_title(graph_title[i])
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel("Offset in z-direction")
    ax.legend()
#ax = fig.add_subplot(2,4,8)
#leg = pl.legend(lines, labels, loc=2)
#ax.add_artist(leg)
fig.tight_layout()
fig.savefig("/home/robbis/Dropbox/PhD/rg_papers/megmri_paper/"+str(i)+"_"+metric+".png", dpi=150)
    

mutual = np.array([[0.916632,0.915068,0.916718],
                   [0.916551,0.915684,0.917098],
                   [0.916832,0.915901,0.916693],
                   [0.917053,0.915852,0.917052],
                   [0.916621,0.916179,0.916621],
                   [0.916604,0.915090,0.916316]]).T
                   
                   
################   PHANTOM   ##########################
values = np.array([0.922231, 0.909037, 0.917880])-0.005
parameters = [6,9,12]           

fig = pl.figure(dpi=150)
pl.plot(parameters, 1/values, lw=2.5, alpha=0.8, markersize=12, marker='o')
pl.title("NMI values for 6, 9, 12 parameter fits")
pl.xticks(parameters, ("Rigid", "Zoom", "Shear"))
pl.xlabel("Transformation type")
pl.ylabel("Normalized Mutual Information")
fig.tight_layout()
fig.savefig("/home/robbis/Dropbox/PhD/rg_papers/megmri_paper/phantom_literal.png", dpi=150)

###############    AALTO  #########################
values = 100/np.array([87.68, 87.71, 87.63, 87.63, 87.56, 87.61])
parameters = range(6)

fig = pl.figure(dpi=150)
pl.plot(parameters, values, lw=2.5, alpha=0.8, markersize=12, marker='o')
pl.title("NMI values for different y-direction offsets")
#pl.xticks(parameters, ("Rigid", "Zoom", "Shear"))
pl.xlabel("Offset in y-direction")
pl.ylabel("Normalized Mutual Information")
pl.ticklabel_format(useOffset=False)
fig.tight_layout()
fig.savefig("/home/robbis/Dropbox/PhD/rg_papers/megmri_paper/aalto.png", dpi=150)



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
  

#############################################
    
file_name = '/home/robbis/development/svn/mutual/trunk/fileList/listaLF_phantom_37ave.txt'

off_x = [1]#np.arange(1)
off_y = np.arange(4)
off_z = np.arange(5)
params = np.array([6, 9, 12])
#params = np.array([12])

prod = itertools.product(off_x, off_y, off_z, params)

for off in prod:
    path = '/home/robbis/development/svn/mutual/trunk/result/aalto_params/'
    fname = 'brain_offset_%s%s%s_%02d' % (str(off[0]),
                                                 str(off[1]),
                                                 str(off[2]),
                                                 off[3])
    directory = os.path.join(path,fname)
    command = 'mkdir -p %s' % (directory)
    print command
    os.system(command)
    
    
    out_name = os.path.join(directory, fname)
    command = './Mutual.x %s %s %s %s %s > %s.txt' % ('foo', str(off[3]), str(off[0]), str(off[1]), str(off[2]), out_name)
    print command
    os.system(command)
    
    command = 'mv result/*.tif %s' % (directory)
    print command
    os.system(command)
    
############################################    
off_x = np.arange(2)
off_y = np.arange(4)
off_z = np.arange(5)
params = np.array([6,9,12])

prod = itertools.product(params, off_x, off_y, off_z)

results = np.zeros((2,4,5,3))

for off in prod:
    path = '/home/robbis/development/svn/mutual/trunk/result/aalto_params/'
    fname = 'brain_offset_%s%s%s_%02d' % (str(off[1]),
                                          str(off[2]),
                                          str(off[3]),
                                          off[0]
                                          )
    
    fp = file(os.path.join(path, fname, fname+'.txt'), 'r')
    lines = fp.readlines()
    value3_ = lines[-3]
    value56_ = lines[-56]
    try:
        results[off[1], off[2], off[3], (off[0]/3-2)] = np.float(value3_)
    except Exception as exc:
        results[off[1], off[2], off[3], (off[0]/3-2)] = np.float(value56_)
        continue
    
    
    
  

#############################################
path = '/home/robbis/phantoms/Coregistration/Comparison/FSL/'
lista_file = os.listdir(path)

for f in lista_file:
    input_ = input_ = os.path.join(path, f)
    output_ = os.path.join(path,'%s_cp.tif' %(f[:-4]))
    convert_ = 'tiffcp -r 1 -c none %s %s' % (input_, output_)
    print(convert_)
    os.system(convert_)


################################################
histogram_init = np.genfromtxt("/home/robbis/phantoms/Coregistration/paper-material/histogram_final.txt")
histogram_fina = np.genfromtxt("/home/robbis/phantoms/Coregistration/paper-material/histogram_001.txt")

histogram_fina[histogram_fina==0] = 1
histogram_init[histogram_init==0] = 1

histogram_fina = np.log10(histogram_fina)
histogram_init = np.log10(histogram_init)

gray_level = 150

metrics_ = []

selected_functions = {name:fx for name,fx in dict_functions.iteritems() if name in selected_keys}
for name, fx in selected_functions.iteritems():
    try:
        m = fx(histogram_fina[:gray_level, :gray_level].flatten(), 
               histogram_init[:gray_level, :gray_level].flatten())
    except Exception as err:
        print (name, err)
        continue
        
    metrics_.append((name, m))

##############################################################
from matplotlib import colors
pl.rcParams.update({'font.size': 20})

fname = "histogram_final.txt"
#fname = "histogram_001.txt"

img = np.genfromtxt("/home/robbis/phantoms/Coregistration/paper-material/"+fname)
img[img==0] = 1
norm = colors.LogNorm()

pl.imshow(img, norm=norm, vmax=100, origin='lower', cmap=pl.cm.nipy_spectral, interpolation='bilinear')

pl.savefig("/home/robbis/phantoms/Coregistration/paper-material/"+fname[:-4]+".png", dpi=300)

##############################################################

def get_offset(px, x):
    if px == x-1:
        return 0.25
    elif px == 0:
        return -0.25
    return 0

def get_axis_offset(offset):
    if offset > 0:
        off_b, off_e = offset, 0
    elif offset < 0:
        off_b, off_e = 0, offset
        
    else:
        off_b, off_e = 0.25*0.5, -0.25*0.5
        
    return off_b, off_e


def plot_projection(fig,
                    index,
                    projection,
                    py,
                    y,
                    y_label,
                    px,
                    x,
                    x_label,
                    min_=0,
                    max_=1,
                    interpolation="bicubic"
                    
                    ):
    ax = fig.add_subplot(2,2,index)
    cax = ax.imshow(projection.T, interpolation=interpolation, cmap=pl.cm.jet, vmin=min_, vmax=max_)
        
    offset = get_offset(py, y)
    ax.hlines(py+offset, 0-0.5, x-0.5)
    ax.set_yticklabels(np.arange(y))
    ax.set_yticks(np.arange(y)+offset)
    ax.set_ylabel("Offset on %s direction" % (y_label))
    off_b, off_e = get_axis_offset(offset)
    ax.set_ylim(y-0.5+2*off_e, 0-0.5+2*off_b)
    print y-0.5+off_e, 0-0.5+off_b
    
        
    offset = get_offset(px, x)
    ax.vlines(px+offset, 0-0.5, y-0.5)
    ax.set_xticklabels(np.arange(x))
    ax.set_xticks(np.arange(x)+offset)
    ax.set_xlabel("Offset on %s direction" % (x_label))
    off_b, off_e = get_axis_offset(offset)
    ax.set_xlim(0-0.5+2*off_b, x-0.5+2*off_e)
    print x-0.5+off_e, 0-0.5+off_b
    



path_la = "/home/robbis/development/svn/mutual/trunk/result/los_alamos/results_x-y-z-par_numpy_array.npz"
path_la = "/home/robbis/development/svn/mutual/trunk/result/aalto/results_x-y-z-par_numpy_array.npz"
results_xyz = np.load(path_la)
results_xyz = results_xyz['arr_0']

xyz = 1/results_xyz[...,1]
xyz = np.rollaxis(xyz, 2, 0)

def norm(x, a=1.105, b=1.085):

    max_ = x.max()
    min_ = x.min()
     
    c = ((b-a)*(x-min_))/(max_-min_)+a
    return c

#xyz_norm = norm(xyz, a=1.142, b=1.138)
xyz_norm = norm(xyz)
max_ = xyz_norm.max()
min_ = xyz_norm.min()

shape_ = xyz_norm.shape

px, py, pz = [1,1,0]
#px, py, pz = [5,1,0]


x_ = shape_[0]-1
y_ = shape_[1]
z_ = shape_[2]

""" LOS ALAMOS
fig = pl.figure(figsize=(12,7))

plot_projection(fig, 1, xyz_norm[:,py,:], pz, z_, "z", px, x_, "x", min_=min_, max_=max_)
plot_projection(fig, 2, xyz_norm[px,:,:], pz, z_, "z", py, y_, "y", min_=min_, max_=max_)  
plot_projection(fig, 3, xyz_norm[:,:,pz], py, y_, "y", px, x_, "x", min_=min_, max_=max_)
"""

fig = pl.figure(figsize=(10,13))

plot_projection(fig, 1, xyz_norm[:4,py,:], px, x_, "x", pz, z_, "z", min_=min_, max_=max_)
plot_projection(fig, 2, xyz_norm[pz,:,:], px, x_, "x", py, y_, "y", min_=min_, max_=max_)  
plot_projection(fig, 3, xyz_norm[:4,:,px], py, y_, "y", pz, z_, "z", min_=min_, max_=max_)
pl.tight_layout()

pl.savefig("/home/robbis/windows-vbox/aalto_2.png", dpi=150)



#######################################################
path = "/home/robbis/development/svn/mutual/trunk/result/los_alamos/brain_offset_000_06/"
path = "/home/robbis/development/svn/mutual/trunk/2015/offset_z_0/"
img_list = os.listdir(path)
pre_  = [i for i in img_list if i.find("1_") != -1]
post_ = [i for i in img_list if i.find("2_") != -1]

pre_.sort()
post_.sort()

pre_stack = []
post_stack = []

for i, img in enumerate(pre_):
    fname_pre = os.path.join(path, img)
    fname_post = os.path.join(path, post_[i])
    im_pre = Image.open(fname_pre)
    im_post = Image.open(fname_post)
    pre_stack.append(np.asanyarray(im_pre))
    post_stack.append(np.asanyarray(im_post))
    
post_stack = np.dstack(post_stack)
pre_stack = np.dstack(pre_stack)

mat = {}
mat['hf'] = post_stack
mat['lf'] = pre_stack

#####################################################################

from sklearn.mixture import GaussianMixture
from scipy import ndimage


def plot_log_histogram(img, means_=None, interp=None, vmax=None):
    from matplotlib import colors
    pl.figure()
    img[img==0] = 1
    norm = colors.LogNorm()

    pl.imshow(img, 
              norm=norm, 
              vmax=vmax, 
              origin='lower', 
              cmap=pl.cm.nipy_spectral, 
              interpolation=interp
              )
    pl.colorbar()
    if means_ != None:
        pl.scatter(means_[:,0],
                   means_[:,1],
                   s=100,
                   c='red')
    


def make_histogram(img1, img2, n_bin=256):
    histogram = np.zeros((n_bin, n_bin))
    for x, y in zip(img1, img2):
        histogram[x,y] += 1
        
    return histogram


def get_images(tiff_list):
    images = []
    for i, img in enumerate(tiff_list):
        img_ = Image.open(img)
        images.append(np.asanyarray(img_))
        
    return np.dstack(images)


def get_file_list(path, pattern, end=None):
    import glob
    flist = glob.glob(os.path.join(path, pattern))
    flist.sort()
    if end == None:
        end=len(flist)
    
    return flist[:end]
    

def reverse_histogram(histogram):
    data = []
    lenx, leny = histogram.shape
    
    for i in range(lenx):
        for j in range(leny):
            n_elem = np.int(np.rint(histogram[i,j]))
            if n_elem > 1:
                datum = np.zeros((n_elem, 2))
                datum[:,:] = i, j
            
                data.append(datum)
    
    return np.vstack(data)



hf_init = get_file_list("/home/robbis/development/svn/mutual/trunk/pre", "*.tif", 6)
hf_final = get_file_list("/home/robbis/development/svn/mutual/trunk/result", "2_*.tif", 6)
lf = get_file_list("/home/robbis/development/svn/mutual/trunk/result", "1_*.tif")

hf_init_img = get_images(hf_init)
hf_final_img = get_images(hf_final)
lf = get_images(lf)


hf_i_flat = hf_init_img.flatten()
hf_f_flat = hf_final_img.flatten()
lf_flat = lf.flatten()


histo_init = make_histogram(lf_flat, hf_i_flat)
histo_final = make_histogram(lf_flat, hf_f_flat)

sm_final = ndimage.gaussian_filter(histo_final, sigma=1.)
sm_init = ndimage.gaussian_filter(histo_init, sigma=1.)

sm_init[sm_init == 0] = 1
sm_final[sm_final == 0] = 1
init = reverse_histogram(np.log10(sm_init))
final = reverse_histogram(np.log10(sm_final))


#init = np.vstack((lf_flat, hf_i_flat)).T
#final = np.vstack((lf_flat, hf_f_flat)).T

m_init = np.array([[  0.,   0.],
                   [ 54.,  10.],
                   [ 80.,  75.]])

gmm = GaussianMixture(n_components=3, means_init=m_init)

gmm.fit(init)
means_i = gmm.means_
cov_i = gmm.covariances_

m_final = np.array([[  0.,   0.],
                   [ 42.,  17.],
                   [ 80.,  75.]])
gmm = GaussianMixture(n_components=3, means_init=m_final)
gmm.fit(final)
means_f = gmm.means_
cov_f = gmm.covariances_

plot_log_histogram(histo_init, means_i, vmax=100, interp="bicubic")
pl.scatter(m_init[:,0], m_init[:,1], s=100, marker="^")
plot_log_histogram(histo_final, means_f, vmax=100, interp="bicubic")
pl.scatter(m_final[:,0], m_final[:,1], s=100, marker="^")



#################################
path = 'development/svn/mutual/trunk/result/los_alamos/brain_offset_000_06/'
image_list = os.listdir(path)
image_list = [i for i in image_list if i.find('2_imTr') != -1]
image_list.sort()
image_ = [np.array(Image.open(os.path.join(path, f))) for f in image_list]
image = np.rollaxis(np.array(image_), 0, 3)
image = np.rollaxis(np.array(image), 0, 2)
image = np.flip(np.flip(image, axis=0), axis=1)
ni.save(ni.Nifti1Image(image[...,:5], affine), 'phantoms/Coregistration/BREAKBEN/lanl/lanl_hf.nii.gz')


# Cluster overlap #
path = "phantoms/Coregistration/BREAKBEN/lanl/"
manual_mask = ni.load(os.path.join(path, "manual_mask_lanl_masked.nii.gz"))
automat_mask = ni.load(os.path.join(path, "clustering_lanl.nii.gz"))

autom_data = automat_mask.get_data()
manual_data = np.int_(manual_mask.get_data())

pl.figure()
pl.imshow(autom_data[...,2])
pl.figure()
pl.imshow(manual_data[...,2])
overlap_matrix = np.zeros((autom_data.max(), manual_data.max()))

aut_mask_nilearn = np.zeros(autom_data.shape + (autom_data.max(),))
man_mask_nilearn = np.zeros(manual_data.shape + (manual_data.max(),))

for aroi in np.unique(autom_data)[1:]:
    amask = autom_data == aroi
    aut_mask_nilearn[...,aroi-1] = np.int_(amask) * (aroi+1)
    
    for mroi in np.unique(manual_data)[1:]:
        
        
        mmask = manual_data == mroi
        man_mask_nilearn[...,mroi-1] = np.int_(mmask) * (mroi+1)
        
        
        overlap = np.count_nonzero(np.logical_and(amask, mmask))
        total = np.count_nonzero(mmask)
        
        overlap_matrix[aroi-1, mroi-1] = overlap / np.float(total)





pl.imshow(overlap_matrix.T, vmin=0, vmax=1.)
pl.yticks(range(4), ['Eye', 'Tissue under the eye', 'White matter', 'Gray and white matter'])
pl.colorbar()
pl.xticks(range(5), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])


ni.save(ni.Nifti1Image(manual_data, automat_mask.affine),
        os.path.join(path, "manual_mask_lanl.nii.gz"))

ni.save(ni.Nifti1Image(aut_mask_nilearn, automat_mask.affine), 
        os.path.join(path, "automatic_mask_lanl_nilearn.nii.gz"))
ni.save(ni.Nifti1Image(man_mask_nilearn, automat_mask.affine), 
        os.path.join(path, "manual_mask_lanl_nilearn.nii.gz"))





hf_data = ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_hf.nii.gz").get_data()
lf_data = ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_lf.nii.gz").get_data()
clusters = np.zeros((autom_data.max(), 2))



for aroi in np.unique(autom_data)[1:]:
    mask_ = autom_data == aroi
    
    clusters[aroi-1, 0] = hf_data[mask_].mean()
    clusters[aroi-1, 1] = lf_data[mask_].mean()
    
    print aroi, hf_data[mask_].mean(), lf_data[mask_].mean()

pl.figure()
pl.imshow(clusters.T, cmap=pl.cm.gray)
pl.yticks(range(2), ['High Field', 'Low Field'])
pl.xticks(range(5), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])

pl.figure()
pl.imshow(np.arange(1, autom_data.max()+1)[np.newaxis,:], cmap=pl.cm.gist_rainbow)
pl.xticks(range(5), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])




# LANL

# Clustering sara
nil.plot_prob_atlas(ni.load(os.path.join(path, "automatic_mask_lanl_nilearn.nii.gz")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_hf.nii.gz"), 
                    view_type='filled_contours',
                    alpha=0.3,
                    draw_cross=False,
                    display_mode='z',
                    threshold=2,
                    annotate=False)

# Manual clustering
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_lanl_nilearn.nii.gz")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_hf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    threshold=None,
                    linewidths=1,
                    annotate=False)

# High-field
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_lanl_nilearn.nii.gz")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_hf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    threshold=50,
                    linewidths=1,
                    annotate=False)


# Low Field
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_lanl_nilearn.nii.gz")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/lanl/lanl_lf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    threshold=10,
                    annotate=False)







### AALTO tiff-to-nifti ###
path = 'phantoms/Coregistration/AALTO/conversion/'
image_list = os.listdir(path)
image_list_lf = [i for i in image_list if i.find('1_imTr') != -1]
image_list_hf = [i for i in image_list if i.find('2_imTr') != -1]
image_list_hf.sort()
image_list_lf.sort()

image_ = [np.array(Image.open(os.path.join(path, f))) for f in image_list_hf]
image = np.rollaxis(np.array(image_), 0, 3)
image = np.rollaxis(np.array(image), 0, 2)
image = np.flip(np.flip(image, axis=0), axis=1)
affine = np.eye(4)
ni.save(ni.Nifti1Image(image[...,2:], affine), 'phantoms/Coregistration/BREAKBEN/aalto_hf.nii.gz')

image_ = [np.array(Image.open(os.path.join(path, f))) for f in image_list_lf]
image = np.rollaxis(np.array(image_), 0, 3)
image = np.rollaxis(np.array(image), 0, 2)
image = np.flip(np.flip(image, axis=0), axis=1)
affine = np.eye(4)
ni.save(ni.Nifti1Image(image[...,2:], affine), 'phantoms/Coregistration/BREAKBEN/aalto_lf.nii.gz')


clust_aalto_mat = loadmat("phantoms/Coregistration/BREAKBEN/ris_clust_aalto.mat")
clust_data = clust_aalto_mat['ris_clust_aalto']
#clust_data = np.rollaxis(clust_data, 0, 3)
clust_data = np.rollaxis(clust_data, 0, 2)
clust_data = np.flip(np.flip(clust_data, axis=0), axis=1)
ni.save(ni.Nifti1Image(clust_data, affine), 'phantoms/Coregistration/BREAKBEN/clustering_aalto.nii.gz')


# Cluster overlap #
path = "phantoms/Coregistration/BREAKBEN/aalto"
manual_mask = ni.load(os.path.join(path, "manual_mask_aalto.img"))
automat_mask = ni.load(os.path.join(path, "clustering_aalto.nii.gz"))

autom_data = automat_mask.get_data()
manual_data = np.flip(manual_mask.get_data(), axis=0)

pl.figure()
pl.imshow(autom_data[...,2])
pl.figure()
pl.imshow(manual_data[...,2])
overlap_matrix = np.zeros((autom_data.max(), manual_data.max()))

aut_mask_nilearn = np.zeros(autom_data.shape + (autom_data.max(),))
man_mask_nilearn = np.zeros(manual_data.shape + (manual_data.max(),))

for aroi in np.unique(autom_data)[1:]:
    amask = autom_data == aroi
    aut_mask_nilearn[...,aroi-1] = np.int_(amask) * (aroi+1)
    
    for mroi in np.unique(manual_data)[1:]:
        
        
        mmask = manual_data == mroi
        man_mask_nilearn[...,mroi-1] = np.int_(mmask) * (mroi+1)
        
        
        overlap = np.count_nonzero(np.logical_and(amask, mmask))
        total = np.count_nonzero(mmask)
        
        overlap_matrix[aroi-1, mroi-1] = overlap / np.float(total)
        
pl.imshow(overlap_matrix.T, vmin=0, vmax=1.)
pl.yticks(range(1), ['Structural Landmark'])
pl.colorbar()
pl.xticks(range(4), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])


ni.save(ni.Nifti1Image(aut_mask_nilearn, automat_mask.affine), 
        os.path.join(path, "automatic_mask_aalto_nilearn.img"))
ni.save(ni.Nifti1Image(man_mask_nilearn, automat_mask.affine), 
        os.path.join(path, "manual_mask_aalto_nilearn.img"))


hf_data = ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_hf.nii.gz").get_data()
lf_data = ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_lf.nii.gz").get_data()
clusters = np.zeros((autom_data.max(),2))
for aroi in np.unique(autom_data)[1:]:
    mask_ = autom_data == aroi
    
    clusters[aroi-1, 0] = hf_data[mask_].mean()
    clusters[aroi-1, 1] = lf_data[mask_].mean()
    
    print aroi, hf_data[mask_].mean(), lf_data[mask_].mean()

pl.figure()
pl.imshow(clusters.T, cmap=pl.cm.gray)
pl.yticks(range(2), ['High Field', 'Low Field'])
pl.xticks(range(4), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

pl.figure()
pl.imshow(np.arange(1, autom_data.max()+1)[np.newaxis,:], cmap=pl.cm.gist_rainbow)
pl.yticks(range(2), ['High Field', 'Low Field'])
pl.xticks(range(4), ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])



## Plot nilearn ##

# Clustering sara
nil.plot_prob_atlas(ni.load(os.path.join(path, "automatic_mask_aalto_nilearn.hdr")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_hf.nii.gz"), 
                    view_type='filled_contours',
                    alpha=0.4,
                    draw_cross=False,
                    display_mode='z',
                    threshold=2,
                    linewidths=1,
                    annotate=False)

# Mask manual
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_aalto_nilearn.hdr")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_hf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    linewidths=1,
                    threshold=None,
                    annotate=False)

# Low field
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_aalto_nilearn.hdr")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_lf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    threshold=10,
                    annotate=False)


# High field
nil.plot_prob_atlas(ni.load(os.path.join(path, "manual_mask_aalto_nilearn.hdr")), 
                    anat_img=ni.load("phantoms/Coregistration/BREAKBEN/aalto/aalto_hf.nii.gz"), 
                    view_type='contours',
                    draw_cross=False,
                    display_mode='z',
                    threshold=10,
                    annotate=False)