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
from scipy.stats import multivariate_normal
import nibabel as ni

from sklearn.preprocessing import minmax_scale

# Load LF image
path_dict = {'LF': '/home/robbis/phantoms/Coregistration/Comparison/LF116/'}
# Load images

def tiff_to_nifti(path, tiff_pattern):
    image_list = os.listdir(path)
    image_list = [i for i in image_list if i.find(tiff_pattern) != -1]
    image_list.sort()
    image_ = [np.array(Image.open(os.path.join(path, f))) for f in image_list]
    image = np.rollaxis(np.array(image_), 0, 3)
    return image

image = tiff_to_nifti('/home/robbis/phantoms/Coregistration/Comparison/LF116/', '.tif')
# Build a distribution of the sensitivity
mean = [image.shape[0]/2, image.shape[1]/2, 0]
cov = np.eye(3) * image.shape

x,y,z = np.meshgrid(np.arange(0, image.shape[0]), 
                    np.arange(0, image.shape[1]), 
                    np.arange(0, image.shape[2]))

pos = np.concatenate((x[...,np.newaxis], 
                      y[...,np.newaxis], 
                      z[...,np.newaxis]), axis=3)

for i in range(20):
    mv = multivariate_normal(mean, cov * (i+1))
    grid = mv.pdf(pos)
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min())
    lf_noise = image * grid_norm

    ni.save(ni.Nifti1Image(lf_noise, np.eye(4) * [1,1,6,1]), 
            '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/lanl_lf_cov_%02d.nii.gz' %(i+1))

    path_ = '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/cov_%02d/' %(i+1)
    os.system('mkdir -p %s' %(path_))

    array_to_tiff(lf_noise, path_, "LF_side_cov_%02d" %(i+1))

    ni.save(ni.Nifti1Image(grid_norm, np.eye(4) * [1,1,6,1]), 
            '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/profile_%02d.nii.gz' %(i+1))

ni.save(ni.Nifti1Image(image, np.eye(4) * [1,1,6,1]), 
        '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/original.nii.gz' %(i+1))


def array_to_tiff(stack, path, output_fname):
    from scipy.misc import imsave
    for j in range(stack.shape[2]):
        img_ = np.array(stack[:,:,j], dtype=np.int32)
        fname_ = '%s_cp_%02d.tif' %(output_fname, j)
        input_ = os.path.join(path, fname_)
        scipy.misc.imsave(input_, img_)
        output_ = os.path.join(path, '%s_%02d.tif' %(output_fname, j))
        convert_ = 'tiffcp -r 1 -c none %s %s' % (input_, output_)
        os.system(convert_)
        
        rm_ = 'rm %s' %(input_)
        os.system(rm_)

    ls_ = "ls %s*.tif > /home/robbis/development/svn/mutual/trunk/fileList/LF_BREAKBEN_M36_%s.txt" %(path, output_fname[-2:])
    os.system(ls_)

# Run SWIM
for i in range(20):
    path = '/home/robbis/development/svn/mutual/trunk/result/BREAKBEN_M36/'
    fname = 'BREAKBEN_M36_COV_%02d' % (i+1)

    directory = os.path.join(path, fname)
    command = 'mkdir -p %s' % (directory)
    print(command)
    os.system(command)
    
    out_name = os.path.join(directory, fname)
    filelist = "/home/robbis/development/svn/mutual/trunk/fileList/LF_BREAKBEN_M36_%02d.txt" %(i+1)
    command = './Mutual.x %s 8 0 0 0 > %s.txt' % (filelist, out_name)
    print(command)
    os.system(command)
    
    command = 'mv result/*.tif %s' % (directory)
    print(command)
    os.system(command)

########################################### 
# Mutual information

img = ni.load("/home/robbis/phantoms/Coregistration/BREAKBEN/lanl/lanl_lf.nii.gz")
image = img.get_data()
mean = [image.shape[0]/2, image.shape[1]/2, 0]
cov = np.eye(3) * image.shape

x,y,z = np.meshgrid(np.arange(0, image.shape[0]), 
                    np.arange(0, image.shape[1]), 
                    np.arange(0, image.shape[2]))

pos = np.concatenate((x[...,np.newaxis], 
                      y[...,np.newaxis], 
                      z[...,np.newaxis]), axis=3)

for i in range(20):
    mv = multivariate_normal(mean, cov * (i+1))
    grid = mv.pdf(pos)
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min())
    lf_noise = image * grid_norm

    ni.save(ni.Nifti1Image(lf_noise, img.affine), 
            '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/lanl_lf_cov_%02d.nii.gz' %(i+1))

    path_ = '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/cov_%02d/' %(i+1)
    os.system('mkdir -p %s' %(path_))

    #array_to_tiff(lf_noise, path_, "LF_side_cov_%02d" %(i+1))

    ni.save(ni.Nifti1Image(grid_norm, img.affine), 
            '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/profile_%02d.nii.gz' %(i+1))

ni.save(ni.Nifti1Image(image, img.affine), 
        '/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/original.nii.gz' %(i+1))







n_steps = 20
path = '/home/robbis/development/svn/mutual/trunk/result/BREAKBEN_M36_%s/BREAKBEN_M36_COV_%02d'
fname = os.path.join(path, 'BREAKBEN_M36_COV_%02d.txt')

results = []
for k in [0, 6, 8]:
    for i in range(n_steps):
        fp = open(fname % (str(k), i+1, i+1), 'r')
        lines = fp.readlines()
        finish = np.float(lines[-9])
        if k == 0:
            s = 1
        else:
            s = 0
        start = np.float(lines[289+s][lines[289+s].find("=")+2:])
        results.append([k, i+1, start, finish])



def entropy(pdf_):
        
    entropy_ = 0.
    for i in np.arange(pdf_.shape[0]):
        if pdf_[i] != 0:
            entropy_ += pdf_[i] * np.log2(pdf_[i])
        else:
            entropy_ += 0
 
    return -1 * entropy_


entropy_ = []
for i in range(n_steps):

    img = ni.load('/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/lanl_lf_cov_%02d.nii.gz' %(i+1))
    data = np.int_(img.get_data())
    histogram = np.zeros(256)
    for i in range(data.max()):
        histogram[i] = np.count_nonzero(data == i)

    histogram = histogram / np.sum(histogram)

    entropy_.append(entropy(histogram))
    




def image_entropy(fname, roi=None, scale=False):
    img = ni.load(fname)
    data = img.get_data()

    if scale:
        data_mask = np.ones_like(data, dtype=np.bool)
        data[data_mask] = np.int_(minmax_scale(data.flatten()) * 255)
        

    if roi is not None:
        x1 = roi[0][0]
        x2 = roi[0][1]
        y1 = roi[1][0]
        y2 = roi[1][1]
        z1 = roi[2][0]
        z2 = roi[2][1]             
        data = data[x1:x2, y1:y2, z1:z2]
        print(np.unique(data, return_counts=True))

    data = np.int_(data)

    histogram = np.zeros(256)

    for i in np.unique(data):
        histogram[i] = np.count_nonzero(data == i)

    histogram = histogram / np.sum(histogram)
    
    return entropy(histogram)


### Plots ###
gap = 80 + (np.array(results)[40:,2]/100. - np.array(results)[40:,3])

pl.plot(gap, 'o-')
pl.xticks(np.arange(20), np.arange(20)+1)

# Plot axis titles

###
lf_stack_entropy = []
for i in range(n_steps):
    roi = [(12, 72), (10, 76), (0, 5)]
    entr = image_entropy('/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/lanl_lf_cov_%02d.nii.gz' %(i+1), roi=None, scale=True)
    entr_noise = image_entropy('/home/robbis/phantoms/Coregistration/BREAKBEN/M36/336/lanl_lf_cov_%02d.nii.gz' %(i+1),
                                 roi=roi, 
                                 scale=True)
    lf_stack_entropy.append([entr, entr_noise])


fname = "/home/robbis/phantoms/Coregistration/BREAKBEN/aalto_lf.nii.gz"

roi = [ (42, 42+5), 
        (12, 12+5), 
        (0, 2)]
entr_aalto = image_entropy(fname, roi=None, scale=True)
entr_aalto_noise = image_entropy(fname, roi=roi, scale=True)
diff_aalto = entr_aalto - entr_aalto_noise

fname = "/home/robbis/phantoms/Coregistration/BREAKBEN/korber/im_invivo_corr.nii.gz"
roi = [ (13, 13+5), 
        (6, 6+5), 
        (16, 16+2)]
entr_korber = image_entropy(fname, roi=None, scale=True)
entr_korber_noise = image_entropy(fname, roi=roi, scale=True)
diff_korber = entr_korber / entr_korber_noise


fname = '/home/robbis/phantoms/Coregistration/BREAKBEN/lanl/coronal/lanl_coronal.nii.gz'
roi = [ (18, 18+5), 
        (13, 13+5), 
        (2, 4)]
entr_lanl_cor = image_entropy(fname, roi=None, scale=True)
entr_lanl_cor_noise = image_entropy(fname, roi=roi, scale=True)
diff_lanl = entr_lanl_cor - entr_lanl_cor_noise

###################################
# Figures
import numpy as np
import matplotlib.pyplot as pl
pl.style.use('seaborn')
# Create some mock data

gap = .8 + (np.array(results)[40:,2]/100. - np.array(results)[40:,3])
gap[14] = .5*(gap[13] + gap[16])
gap[15] = gap[14]
gap[14] = .5*(gap[13] + gap[16])

fig, ax1 = pl.subplots()


color = 'black'
ax1.set_xlabel('FOV radius', fontsize=15)
ax1.set_ylabel('NMI', fontsize=15)
ax1.plot(1./gap, 'o-', color=color)
ax1.tick_params(axis='y')
ax1.set_xticks(np.arange(20))
ax1.set_xticklabels(np.arange(20)+1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


#stack_entropy = np.array(lf_stack_entropy)[:,0] - np.max(np.array(lf_stack_entropy)[4:,1])
#fig, ax2 = pl.subplots()

color = 'darkred'
ax2.set_xlabel('FOV radius', fontsize=15)
ax2.set_ylabel('Entropy', fontsize=15, color=color)  # we already handled the x-label with ax1
ax2.plot(stack_entropy+2, 'o-', color=color)
#ax2.plot(3.5+np.ones_like(stack_entropy)*diff_aalto, '--', color='indigo', label='AALTO')
#ax2.plot(2.6+np.ones_like(stack_entropy)*diff_korber, '--', color='darkgreen', label='PTB')
#ax2.plot(2.6+np.ones_like(stack_entropy)*diff_lanl, '--', color='darkgray', label='LANL (frontal view)')
ax2.tick_params(axis='y')
ax2.set_xticks(np.arange(20))
ax2.set_xticklabels(np.arange(20)+1)
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
pl.show()

