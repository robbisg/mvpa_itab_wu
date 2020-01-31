import nibabel as ni
from scipy.ndimage.interpolation import rotate



for angle in np.arange(40, 90, 10):
    
    img_rot = rotate(image_, angle=angle, axes=(0,2), reshape=False)
    rot_img = ni.Nifti1Image(img_rot, np.eye(4))
    ni.save(rot_img, os.path.join(path, "img_rot_%s.nii.gz" % (str(angle))))
    
    
    

for i in range(6):
    filename = "/home/robbis/development/svn/mutual/trunk/fileList/lista_LANL_LF_Coronal.txt"
    command = "./Mutual.x "+filename+" 9 0 0 %s > /home/robbis/development/svn/mutual/trunk/output.txt" % (str(i))
    print command
    
    command = "mkdir -p /home/robbis/development/svn/mutual/trunk/result/lanl_coronal_%s" %(str(i))
    print command
    
    command = "mv /home/robbis/development/svn/mutual/trunk/result/1_* /home/robbis/development/svn/mutual/trunk/result/lanl_coronal_%s" %(str(i))
    print command
    
    command = "mv /home/robbis/development/svn/mutual/trunk/result/2_* /home/robbis/development/svn/mutual/trunk/result/lanl_coronal_%s" %(str(i))
    print command
    
    command = "mv /home/robbis/development/svn/mutual/trunk/output.txt /home/robbis/development/svn/mutual/trunk/result/lanl_coronal_%s" %(str(i))
    print command