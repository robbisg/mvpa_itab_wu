from PIL import Image
from scipy.ndimage.interpolation import zoom, rotate, shift, affine_transform
from scipy.misc import imsave
import tqdm

imgfile = file('/home/robbis/development/svn/mutual/trunk/fileList/listaBrain3TSag_balanced.txt', 'r')
lista_file = imgfile.readlines()

image = [np.array(Image.open(f.rstrip('\n'))) for f in lista_file]
image = np.array(image)


img = np.zeros_like(image)


patht = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/transformed/"
pathp = "//media/robbis/DATA/fmri/megmri/HF_LANL_STATS/permuted/"
from tqdm import tqdm

n_permutation = 100
for i in tqdm(range(n_permutation)):
    
    # crea matrice transform
    rzoom = 0.2*(np.random.rand(3) - .5) + 1. # [0.5, 1.5)
    #rzoom = [1, 1, 1,]
    rotation_deg = 0.3 * (np.random.rand(3) - 0.5)
    #rotation_deg = [0, 0, 0.5]
    rroto = get_affine_rotation(*rotation_deg)
    
    rshift = 10 * (np.random.rand(3) - 0.5)
    
    # applica transofrm
    cdm = np.array(image.shape)/2
    offset = cdm - np.dot(rroto, cdm)
    img = affine_transform(image, rroto, offset=offset, order=1)
    img = shift(img, rshift, order=1)
    img = zoom(img, rzoom, order=1)
    img1 = pad(img, image)
    img2 = crop(img1, image)
    
    # salva
    save_tiff(patht, 'transformed.tiff', img2, i)
    conf = np.vstack((rzoom, rotation_deg, rshift))
    fname_ = "parameters.txt"
    input_ = os.path.join(patht, str(i+1), fname_)
    np.savetxt(input_, conf, fmt="%f")
    
    # permuta indici
    imgp = np.random.permutation(image.flatten()).reshape(image.shape)
    
    # salva
    save_tiff(pathp, 'permuted.tiff', imgp, i)


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
    
    
    

def save_tiff(path, fname, image, index):
    
    command = 'mkdir -p %s' % (os.path.join(path, str(index+1)))
    #print command
    os.system(command)
    
    filelist = []
    
    for i in range(image.shape[0]):
        fname_ = "%03d_"+fname
        input_ = os.path.join(path, str(index+1), fname_ %(i+1))
        imsave(input_, image[i])
        
        fname_out = fname_[:-5]+"_rc.tiff"
        output_ = os.path.join(path, str(index+1), fname_out % (i+1))
    
        convert_ = 'tiffcp -r 1 -c none %s %s' % (input_, output_)
        #print convert_
        os.system(convert_)
        
        remove_ = "rm %s" % (input_)
        #print remove_
        os.system(remove_)
        
        filelist.append(output_)
    
    
    lfile = file(os.path.join(path, "fileList_"+fname[:-5]+"_%04d.txt" % (index+1)), 'w')
    for item in filelist:
        lfile.write("%s\n" % item)
    lfile.close()
    


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
    
    
    
    
    
def permute():

    
    filedir = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/transformed/"
    #filedir = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/permuted/"
    filelist = os.listdir(filedir)
    filelist = [f for f in filelist if f.find('.txt') != -1]
    filelist.sort()
    
    offset_x = np.arange(3)
    offset_y = np.arange(3)
    offset_z = np.arange(6)
    params = np.array([9])
    
    prod = itertools.product(filelist, offset_x[:1], offset_y[:1], offset_z[:1], params)
    
    p_counter = 0
    for _ in prod:
        p_counter += 1
        
    prod = itertools.product(filelist, offset_x[:1], offset_y[:1], offset_z[:1], params)
    
    for var in prod:
        input_fname, off_x, off_y, off_z, n_par = list(var)
        
        dir_ = 'permutation_%s%s%s_%02d' % (str(off_x),
                                            str(off_y),
                                            str(off_z),
                                            n_par)
        
        n_perm = int(input_fname[-8:-4])
        
        directory = os.path.join(filedir, str(n_perm), dir_)

        command = 'mkdir -p %s' % (directory)
        print command
        os.system(command)
        
        
        out_name = os.path.join(directory, dir_)
        
        
        fname_ = os.path.join(filedir, input_fname)
        
        command = './Mutual.x %s %s %s %s %s > %s.txt' % (fname_, 
                                                          str(n_par), 
                                                          str(off_x), 
                                                          str(off_y), 
                                                          str(off_z), 
                                                          out_name)
        print command
        os.system(command)
        
        command = 'mv result/*.tif %s' % (directory)
        print command
        os.system(command)
    

def read_results():
    
    
    path = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/transformed/"
    #path = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/permuted/"
    
    
    n_permutation = 100
    
    offset_x = np.arange(3)
    offset_y = np.arange(3)
    offset_z = np.arange(6)
    params = np.array([9])
    
    prod = itertools.product(np.arange(n_permutation) + 1, offset_x[:1], offset_y[:1], offset_z[:1], params)
    
    p_counter = 0
    for _ in prod:
        p_counter += 1
        
    prod = itertools.product(np.arange(n_permutation) + 1, offset_x[:1], offset_y[:1], offset_z[:1], params)


    results = np.zeros((n_permutation, 1, 1, 1, 1))
    
    
    
    for off in prod:
        
        fname = 'permutation_%s%s%s_%02d' % (str(off[1]),
                                              str(off[2]),
                                              str(off[3]),
                                              off[4]
                                              )
        
        fp = file(os.path.join(path, str(off[0]), fname, fname+'.txt'), 'r')
        lines = fp.readlines()
        
        position = -3
        if lines[position] == '\n':
            position = 722
        
        if lines[position] == '\n':
            position += 3
            
        results[off[0]-1, 0, 0, 0, 0] = np.float(lines[position][:-1])
        




def read_configuration():
    
    path = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/transformed/"
        
    n_permutation = 100
    
    offset_x = np.arange(3)
    offset_y = np.arange(3)
    offset_z = np.arange(6)
    params = np.array([9])
    
    prod = itertools.product(np.arange(n_permutation) + 1, offset_x[:1], offset_y[:1], offset_z[:1], params)
    
    p_counter = 0
    for _ in prod:
        p_counter += 1
        
    prod = itertools.product(np.arange(n_permutation) + 1, offset_x[:1], offset_y[:1], offset_z[:1], params)


    configuration = np.zeros((n_permutation, params[0]))
    parameters = np.zeros((n_permutation, params[0]))
    
    
    for off in prod:
        
        fname = 'permutation_%s%s%s_%02d' % (str(off[1]),
                                             str(off[2]),
                                              str(off[3]),
                                              off[4]
                                              )
        
        fp = file(os.path.join(path, str(off[0]), fname, fname+'.txt'), 'r')
        
        par = np.genfromtxt(os.path.join(path, str(off[0]), "parameters.txt"))
        #print par
        parameters[off[0]-1] = par.flatten()
        
        lines = fp.readlines()
        
        
        position = -3
        if lines[position] == '\n':
            position = 722
        
        if lines[position] == '\n':
            position += 3
        
        begin = position - 16
        
        """
        if lines[begin][6:7] != '0':
            begin = 720-16-3
        """
            
        for i in range(off[4]):
            configuration[off[0]-1, i] = float(lines[begin+i][-10:])






def transformation(pt, z, a, t):
    t1 = get_affine_rotation(*a) # rotation matrix
    t2 = np.eye(3) * z # zoom matrix
    
    tt = np.dot(t1, t2) # rotation + zoom matrix
    pt1 = np.dot(pt, tt) + t # point transformation
    ptz = pt1 # resolution
        
    return ptz, tt




def get_error():
    
    n_permutation = 100
    
    point = np.array([1., 1., 1.])
    resolution = [3, 3, 6]
    
    point_cdm = point / 2 * resolution
    
    points_transformed = np.zeros((n_permutation, 3))
    fsl_error = np.zeros(n_permutation)
    
    
    radius = euclidean(np.zeros(3), point_cdm)
    
    zibest, aibest, tibest = parameters[96].reshape((3,3)) 
    tbest, abest, zbest = configuration[96].reshape((3,3))
    
    pti, mi = transformation(point_cdm, zibest, aibest, tibest)
    ptbest, mb = transformation(pti, zbest, abest, tbest)
    
    m0 = np.dot(mi, mb)
    
    
    for i in range(n_permutation):
        
        zi, ai, ti = parameters[i].reshape((3,3)) # initial transf
        tb, ab, zb = configuration[i].reshape((3,3)) # best transf
        
        pti, mi = transformation(point_cdm, zi, ai, ti)
        ptb, mb = transformation(pti, zb, ab, tb)
                
        m = np.dot(mi, mb)
        t = (ptb - ptbest)/resolution
        
        error_ = 1./5 * radius**2 * np.trace(np.dot(m.T, m)) + np.dot(t.T, t)
        
        err = np.sqrt(error_)
        
        fsl_error[i] = err
        points_transformed[i] = ptb
        
        
        
    
def flirt():
    
    filedir = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/transformed/"
    #filedir = "/media/robbis/DATA/fmri/megmri/HF_LANL_STATS/permuted/"
    filelist = os.listdir(filedir)
    filelist = [f for f in filelist if f.find('.txt') != -1]
    filelist.sort()
    
    lf = "/home/robbis/phantoms/Coregistration/results/Stats/lf_116.nii.gz"
    
    configuration_fsl = np.zeros((n_permutation, 4, 4))
    
    for i, f in enumerate(filelist):
        img_files = np.loadtxt(os.path.join(filedir, f), dtype=np.str)
        #image = [np.array(Image.open(img.rstrip('\n'))) for img in img_files]
        #image = np.array(image).T
        fname = os.path.join(os.path.join(filedir, str(i+1), "hf_transformed.nii.gz"))
        #ni.save(ni.Nifti1Image(image, affine), fname)
        
        omat = os.path.join(os.path.join(filedir, str(i+1), "fsl_matrix.txt"))
        ofile = os.path.join(os.path.join(filedir, str(i+1), "fsl_solution.nii.gz"))
        command = "flirt -in %s -ref %s -omat %s -out %s -dof 9" %(fname, lf, omat, ofile)
        #print command
        
        #os.system(command)
        configuration_fsl[i] = np.loadtxt(omat)
        
        
        
    

def flirt_error():
    
    n_permutation = 100
    
    point = np.array([1., 1., 1.])
    resolution = [1, 1, 6]
    
    point_cdm = point / 2 * resolution
    
    points_transformed = np.zeros((n_permutation, 3))
    fsl_error = np.zeros(n_permutation)
    
    
    radius = euclidean(np.zeros(3), point_cdm)
    
    zibest, aibest, tibest = parameters[96].reshape((3,3)) 
    tbest, abest, zbest = configuration[96].reshape((3,3))
    
    pti, mi = transformation(point_cdm, zibest, aibest, tibest)
    ptbest, mbest = transformation(pti, zbest, abest, tbest)
    
    m0 = np.dot(mi, mbest)
    
    
    for i in range(n_permutation):
        
        zi, ai, ti = parameters[i].reshape((3,3)) # initial transf
        mb = configuration_fsl[i] # best transf
        
        pti, mi = transformation(point_cdm, zi, ai, ti)
                
        ptb = np.dot(pti, mb[:3,:3])
        
        m = np.dot(mi, mb[:3,:3])
        t = (ptb  - ptbest) / np.array([1.,1.,6.])
        
        error_ = 1./5 * radius**2 * np.trace(np.dot(m.T, m)) + np.dot(t.T, t)
        
        err = np.sqrt(error_)
        
        fsl_error[i] = err
        points_transformed[i] = ptb
        
    return fsl_error, points_transformed
    
    
def plot_error():
    pal = sns.diverging_palette(240, 10, n=2)
    g = sns.FacetGrid(data, row="label", hue="label", aspect=15, size=.5, palette=pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "data", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "data", clip_on=False, color="w", lw=2, bw=.2)
    g.map(pl.axhline, y=0, lw=2, clip_on=False)
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = pl.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, 
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "data")
    
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=.25)
    
    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    
    
    
####### resampling ######
fname_ = "%03d_"+fname
input_ = os.path.join(path, str(index+1), fname_ %(i+1))
imsave(input_, image[i])
# Load images
images_ = dict()
for k, path in path_dict.iteritems():
    file_list = os.listdir(path)
    file_list.sort()
    images_[k] = [np.array(Image.open(os.path.join(path, f))) for f in file_list]
    images_[k] = np.array(images_[k])

key = 'FSL'
for i, img in enumerate(images_[key]):
    zimg336 = zoom(img, 256/951. * 86/256.)
    zimg116 = zoom(img, 256/951.)
    input_ = "/home/robbis/Dropbox/PhD/rg_papers/megmri_paper/Submission_PlosONE/Revision/data/"
    #imsave(input_+key+"_"+str(i+1)+".png", zimg)
    imsave(input_+key+"_"+str(i+1)+"_upsampled_bw.tif", zimg116 > 26)
    imsave(input_+key+"_"+str(i+1)+"_bw.tif", zimg336 > 26)
    
    