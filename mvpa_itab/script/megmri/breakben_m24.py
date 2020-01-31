import nibabel as ni
import numpy as np
from PIL import Image
import matplotlib.pyplot as pl
import os
import nilearn.plotting as nil
from scipy.ndimage.interpolation import shift, affine_transform, zoom



def convert_clustering(mat_file, fname_output):
    from scipy.io import loadmat
    
    clust_aalto_mat = loadmat(mat_file)
    clust_data = clust_aalto_mat['ris_clust_aalto']
    #clust_data = np.rollaxis(clust_data, 0, 3)
    clust_data = np.rollaxis(clust_data, 0, 2)
    clust_data = np.flip(np.flip(clust_data, axis=0), axis=1)
    ni.save(ni.Nifti1Image(clust_data, np.eye(4)), fname_output)


def tiff_to_nifti(path, tiff_pattern, output):
    #path = 'development/svn/mutual/trunk/result/los_alamos/brain_offset_000_06/'
    image_list = os.listdir(path)
    image_list = [i for i in image_list if i.find(tiff_pattern) != -1]
    image_list.sort()
    image_ = [np.array(Image.open(os.path.join(path, f))) for f in image_list]
    image = np.rollaxis(np.array(image_), 0, 3)
    image = np.rollaxis(np.array(image), 0, 2)
    image = np.flip(np.flip(image, axis=0), axis=1)
    #ni.save(ni.Nifti1Image(image, np.eye(4)), output)
    return image

def nifti_to_tiff(ni_fname, tiff_pattern, output):
    img = ni.load(ni_fname)
    data = img.get_data()



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





#############################

def pad(image1, image2):
    
    def check_pad(pad):
        if pad < 0:
            return 0
        else:
            return pad
    
    xpad = np.int(np.ceil(check_pad(image2.shape[0] - image1.shape[0]) * 0.5))
    ypad = np.int(np.ceil(check_pad(image2.shape[1] - image1.shape[1]) * 0.5))
    zpad = np.int(np.ceil(check_pad(image2.shape[2] - image1.shape[2]) * 0.5))
    
    return np.pad(image1, ((xpad,xpad), (ypad,ypad), (zpad,zpad)), 'constant')
    



def crop(image1, image2):
    
    def floor(n1, n2):
        return np.int(np.floor(0.5*(n1 - n2)))
    
    def ceil(n1, n2):
        a = np.int(np.ceil(0.5*(n1 - n2)))
        return -a if a != 0 else None
    
    x,y,z = image1.shape
    xx,yy,zz = image2.shape
    
    return image1[floor(x,xx):ceil(x,xx), 
                  floor(y,yy):ceil(y,yy),
                  floor(z,zz):ceil(z,zz)]




def matrix_ZYX(alpha, phi, theta):
       
    cos = np.cos
    sin = np.sin
    
    phi = -1*phi

    matrix = np.zeros((3,3))
    
    matrix[0,0] = cos(phi)*cos(theta)
    matrix[0,1] = sin(theta)*cos(phi)
    matrix[0,2] =-sin(phi)
    
    matrix[1,0] = sin(alpha)*sin(phi)*cos(theta) - cos(alpha)*sin(theta)
    matrix[1,1] = sin(alpha)*sin(phi)*sin(theta) + cos(alpha)*cos(theta)
    matrix[1,2] = sin(alpha)*cos(phi)
    
    matrix[2,0] = cos(alpha)*sin(phi)*cos(theta) + sin(alpha)*sin(theta)
    matrix[2,1] = cos(alpha)*sin(phi)*sin(theta) - sin(alpha)*cos(theta)
    matrix[2,2] = cos(alpha)*cos(phi)
    
    return matrix



def matrix_XYZ (alpha, phi, theta):
  
    cos = np.cos
    sin = np.sin

    alpha = -1 * alpha
    theta = -1 * theta

    matrix = np.zeros((3,3))
   
    matrix[0,0] = cos(theta)*cos(phi)
    matrix[0,1] = sin(alpha)*sin(phi)*cos(theta) + cos(alpha)*sin(theta)
    matrix[0,2] =-cos(alpha)*sin(phi)*cos(theta) + sin(alpha)*sin(theta)
    
    matrix[1,0] =-sin(theta)*cos(phi)
    matrix[1,1] =-sin(alpha)*sin(phi)*sin(theta) + cos(alpha)*cos(theta)
    matrix[1,2] = cos(alpha)*sin(phi)*sin(theta) + sin(alpha)*cos(theta)
    
    matrix[2,0] = sin(phi)
    matrix[2,1] =-sin(alpha)*cos(phi)
    matrix[2,2] = cos(alpha)*cos(phi)
    
    return matrix




def get_affine_rotation(phi, psi, theta):
    
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    
        
    c_the = np.cos(theta)
    s_the = np.sin(theta)
    
    matrix = np.zeros((3,3))
    
    matrix[0,0] = c_phi * c_psi - c_the * s_phi * s_psi
    matrix[0,1] = s_phi * c_psi + c_the * c_phi * s_psi
    matrix[0,2] = s_phi * s_the
    
    matrix[1,0] = -c_the * s_psi - c_the * s_phi * c_psi
    matrix[1,1] = -s_phi * s_psi + c_the * c_phi * c_psi
    matrix[1,2] = s_the * c_psi
    
    matrix[2,0] = s_the * s_phi
    matrix[2,1] = -s_the * c_phi
    matrix[2,2] = c_the
       
    

    return matrix



    
# crea matrice transform
img = ni.load("/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/Nifti_Analyze/lanl_hf_resampled.nii.gz")

image = img.get_data()
affine = img.affine

rzoom = [1, 1, 1]


rotation_deg = np.array([-0.345291, -.159902, .173314])
rotation_deg = np.array([-0.334467, -0.142755, 0.191307])
rotation_deg = rotation_deg[[2,1,0]]

rroto = matrix_ZYX(*rotation_deg)

#rshift = [-15.25647, 6.53437, -6.38453]
rshift = np.zeros(3)

# applica transofrm
cdm = np.array(image.shape)/2

#cdm = np.array([150,210,220])*0.5

#cdm = np.array([84.314, 118.5615, 152.77])

offset = cdm - np.dot(rroto, cdm)
img = affine_transform(image, rroto, offset=offset, order=1)
img = shift(img, rshift, order=1)
img = zoom(img, rzoom, order=1)
img1 = pad(img, image)
img2 = crop(img1, image)


ni.save(ni.Nifti1Image(img2, affine), "/home/robbis/phantoms/BRAIN/3T/sagittal/balanced/rotated.nii.gz")








