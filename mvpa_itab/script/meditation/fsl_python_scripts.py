import os
from mne.commands.mne_bti2fiff import out_fname
subjlist = []

################################
path = '/media/robbis/DATA/fmri/monks'
dlist = os.listdir(path)
dlist = [d for d in dlist if d.find('.') == -1 and d.find('_') == -1]
mlist = os.listdir('/home/robbis/Share/Gianni_NEW/analyze/')
clist = os.listdir('/home/robbis/Share/Gianni_NEW/CONTROLLI/analyze/')

fmerge = open(path+'/merge.sh', 'w')

for d in dlist:
    name = d[6:]
    
    slist = [m for m in mlist if m.find(name) != -1 and m.find('.hdr') != -1]
    path_f = '/home/robbis/Share/Gianni_NEW/analyze/'
    if len(slist) == 0:
        slist = [c for c in clist if c.find(name) != -1 and c.find('.hdr') != -1]
        path_f = '/home/robbis/Share/Gianni_NEW/CONTROLLI/analyze/'
    slist.sort()
    
    slist_anat = slist[:1]
    slist_bold = slist[1:]
    
    path__ = ' '+path_f
    
    command = 'fslmerge -t '+os.path.join(path,d,'fmri','bold.nii.gz ')+path__+\
            path__.join(slist_bold)
            
    fmerge.write(command)
    fmerge.write('\n')
        
    image = path_f+slist_anat[0]
    nii = os.path.join(path,d,'mprage','mprage.nii.gz ')
    command = 'fslchfiletype NIFTI_GZ '+ image + ' ' + nii 
    
    fmerge.write(command)
    fmerge.write('\n')
    
fmerge.close()
    
for d in dlist:
    
    command = 'rm '+os.path.join(path, d, 'fmri','*')
    print command 
    command = 'rm '+os.path.join(path, d, 'mprage','*')
    print command




####################### HEADER STUFF ########################################


for d in dlist:
    sub_dir = os.path.join(path, d)
    
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))

    struct_list = [m for m in s_list if m.find('mprage') != -1]
    
    image = os.path.join(sub_dir, 'mprage', struct_list[0])    

    command = 'fslswapdim '+ image + ' -z -x -y '+ os.path.join(sub_dir, 'mprage','mprage_orient')
    
    print command
    
    command = 'fslorient -setqformcode 1 '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')
    print command
    
    s_list = os.listdir(os.path.join(sub_dir, 'fmri'))
    
    struct_list = [m for m in s_list if m.find('x') == -1 and m.find('bold.nii.gz') != -1]

    image = os.path.join(sub_dir, 'fmri', 'bold.nii.gz')        

    command = 'fslswapdim '+ image + ' -z -x -y ' + image.strip('.nii.gz')+'_orient'
    print command
    
    command = 'fslorient -setqformcode 1 ' + image.strip('.nii.gz')+'_orient'
    print command


for d in dlist:
    sub_dir = os.path.join(path, d)
        
    command = 'fslview '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')+' &'
    
    print command

    s_list = os.listdir(os.path.join(sub_dir, 'fmri'))
    image = os.path.join(sub_dir, 'fmri', 'bold.nii.gz')        
    command = 'fslview ' + image.strip('.nii.gz')+'_orient'
    
    print command   

for d in dlist[-4:]:
    sub_dir = os.path.join(path, d)
        
    command = 'gzip -d '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')
    print command
    
    command = 'nifti_tool -mod_hdr -mod_field pixdim \'1 1.2 0.9375 0.9375 1 1 1 1\' -infiles '+\
                os.path.join(sub_dir, 'mprage', 'mprage_orient.nii')+\
                ' -prefix '+os.path.join(sub_dir, 'mprage', 'mprage_voxel')
    print command

    command = 'gzip '+os.path.join(sub_dir, 'mprage', 'mprage_voxel.nii')
    
    print command  
    
    command = 'rm '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii')
    print command
    
    command = 'mv '+os.path.join(sub_dir, 'mprage', 'mprage_voxel.nii.gz')+' '+\
                os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')
    print command
    
     
    command = 'flirt -in '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')+\
    ' -out '+os.path.join(sub_dir, 'mprage', 'mprage_orient_111.nii.gz')+\
    ' -ref '+  os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')+\
    ' -applyisoxfm 1'
    
    print command
    
############### REGISTRATION + WARPING STRUCTURAL ################################

join = os.path.join
path_templates = '/media/DATA/fmri/templates_MNI_3mm/'

for s in subjlist:
    
    sub_dir = os.path.join(path, s)
    
    s_list = os.listdir(join(sub_dir, 'mprage'))

    struct_list = [m for m in s_list if m.find('mprage_orient.nii.gz') != -1]

    image = os.path.join(sub_dir, 'mprage', struct_list[0])
    
    # Register structural to mni template
    command = 'flirt -in '+ image + \
                    ' -ref '+os.path.join(path_templates, 'MNI152_T1_3mm.nii.gz') + \
                    ' -omat '+os.path.join(sub_dir, 'mprage','mprage2mni_3mm.mat') + \
                    ' -o '+ os.path.join(sub_dir, 'mprage','mprage2mni_3mm.nii.gz')
                    

    print command
    
    # Linear warping
    command = 'fnirt --in='+ image + \
                    ' --ref=/media/DATA/fmri/templates_MNI_3mm/MNI152_T1_3mm.nii.gz' + \
                    ' --aff='+os.path.join(sub_dir, 'mprage','mprage2mni_3mm.mat') + \
                    ' --iout='+ os.path.join(sub_dir, 'mprage','mprage_fnirt_3mm.nii.gz') + \
                    ' --config=/media/DATA/fmri/monks/fnirt_3mm.cnf'

    print command
###################################################################################
for s in subjlist:
    sub_dir = os.path.join(path, s)
    
    command = 'overlay 1 0 ' +os.path.join('/media/DATA/fmri/MNI152_T1_3mm.nii.gz') + ' 0.000000 8000 '+ \
                 os.path.join(sub_dir, 'fmri', 'bold_mni_3mm.nii.gz')+' 100 1000 '+\
                  os.path.join(sub_dir, 'fmri', 'overlayed_mni.nii.gz')

    print command

#' -init /media/DATA/fmri/monks/070222andzap/fmri/andzap_mprage_to_bold_no.mat' + \
for s in subjlist:
    sub_dir = os.path.join(path, s)    
    
    command = 'flirt -ref '+ os.path.join('/media/DATA/fmri/MNI152_T1_3mm_brain.nii.gz') + \
               ' -in ' + os.path.join(sub_dir, 'fmri', 'bold_orient.nii.gz') + \
               ' -omat ' + os.path.join(sub_dir, 'fmri', s[6:]+'_bold2mni.mat') + \
               ' -o ' + os.path.join(sub_dir, 'fmri', 'bold2mni_flirt.nii.gz') + \
               ' -cost normmi'
    print command


for s in subjlist_bad:
    sub_dir = os.path.join(path, s)
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))
    struct_list = [m for m in s_list if m.find('x') == -1 and m.find('hdr') != -1]

    image = os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')

    command = 'flirt -ref '+ image + \
                    ' -in '+ os.path.join(sub_dir, 'fmri','bold_orient.nii.gz') + \
                    ' -omat '+os.path.join(sub_dir, 'fmri', s[6:]+'_mprage_to_bold.mat') + \
                    ' -o '+ os.path.join(sub_dir, 'fmri','bold_to_mprage_3mm.nii.gz') + \
                    ' -finesearch 5 -dof 6' 

    print command


for s in subjlist:
    sub_dir = os.path.join(path, s)
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))
    struct_list = [m for m in s_list if m.find('x') == -1 and m.find('hdr') != -1]

    image = os.path.join(sub_dir, 'mprage', struct_list[0])
    nii = os.path.join(sub_dir, 'mprage', 'mprage')
    command = 'fslchfiletype NIFTI_GZ '+ image + ' ' + nii 

    print command 

    
for s in subjlist:                                                                     
    sub_dir = os.path.join(path, s)
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))
    struct_list = [m for m in s_list if m.find('x') == -1 and m.find('hdr') != -1]    
    

    struct_3mm = os.path.join(sub_dir, 'mprage',struct_list[0].strip('.hdr')+'_3mm')
    
    
    command = 'flirt -nosearch -ref '+ struct_3mm + \
                       ' -in '+ os.path.join(sub_dir, 'fmri','bold_orient.nii.gz') + \
                       ' -omat '+os.path.join(sub_dir, 'fmri', s[6:]+'_bold_to_mprage.mat') + \
                       ' -o '+ os.path.join(sub_dir, 'fmri','bold_flirt.nii.gz')
    
    
    #print command
    
    bold = os.path.join(sub_dir, 'fmri','bold_orient.nii.gz')
    
    matrix = os.path.join(sub_dir, 'fmri', s[6:]+'_mprage_to_bold.mat')
    
    struct_list = [m for m in s_list if m.find('x') == -1 and m.find('warpcoef') != -1]
    
    command = 'fsl4.1-applywarp --ref=/media/DATA/fmri/templates_MNI_3mm/MNI152_T1_3mm'+ \
                                 '  --in='+ bold + \
                                 '  --premat='+ matrix + \
                                 '  --mask=/media/DATA/fmri/templates_MNI_3mm/MNI152_T1_3mm_brain_mask.nii.gz' + \
                                 '  --out='+ os.path.join(sub_dir, 'fmri', 'bold_mni_3mm_new') + \
                                 '  --warp='+os.path.join(sub_dir, 'mprage', 'mprage_orient_warpcoef.gz.nii.gz') 
                                 
    #print '\n'
    print command


for s in subjlist:
    sub_dir = os.path.join(path, s)
    bold = os.path.join(sub_dir, 'fmri','bold_orient.nii.gz')
    
    
    command = 'fslroi '+bold+' '+bold.strip('.nii.gz')+'_single 0 1'
    print command

command = 'slicesdir -o '
for s in subjlist:
    sub_dir = os.path.join(path, s)
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))
    struct_list = [m for m in s_list if m.find('mni.nii.gz') != -1]
    
    fmri = os.path.join(sub_dir, 'mprage', 'mprage_orient_brain_g.nii.gz')
    #mprage = os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')
    command += fmri+' '+os.path.join(sub_dir, 'mprage', 'mprage_orient.nii.gz')+' '


list_files = []
for s in subjlist:
    sub_dir = os.path.join(path, s)
    s_list = os.listdir(os.path.join(sub_dir, 'mprage'))
    struct_list = [m for m in s_list if m.find('hdr') == -1 
                   and m.find('mprage_orient') == -1 and m.find('.img') == -1]
    list_files.append(os.path.join(sub_dir, 'fmri',struct_list[0]))
    for f in struct_list:
        print 'rm '+os.path.join(sub_dir, 'mprage', f)    




########################################
net_path = '/media/DATA/fmri/templates_fcmri'
network_list = os.listdir('/media/DATA/fmri/templates_fcmri')
network_list = [s for s in network_list if s.find('.') == -1]



for f in network_list:
    
    sub_folder = os.listdir(os.path.join(net_path, f))
    sub_folder = [g for g in sub_folder if g.find('.') == -1]
    
    main_net = ni.load(os.path.join(net_path, f, f+'.nii.gz'))
    
    data = np.zeros_like(main_net.get_data())
    
    for g in sub_folder:
        filenet = os.listdir(os.path.join(net_path,  f, g))[0]
        network_file = os.path.join(net_path, f, g, os.listdir(os.path.join(net_path, f, g))[0])
        img = ni.load(network_file)
        
        data += 2*img.get_data()*np.float(filenet.strip('.nii.gz'))
    
    ni.save(ni.Nifti1Image(data, img.get_affine(), header=img.get_header()), 
            os.path.join(net_path, f, f+'_separated.nii.gz'))
    
    file_new = os.path.join(net_path, f, f+'_separated.nii.gz')
    file_old = os.path.join(net_path, f, f+'.nii.gz')
    
    command = 'flirt -in '+file_new+' -out '+file_new.strip('.nii.gz')+\
    '_3mm.nii.gz -ref /media/DATA/fmri/MNI152_T1_3mm -applyisoxfm 3 -interp nearestneighbour'
    print command
    
    command = 'flirt -in '+file_old+' -out '+file_old.strip('.nii.gz')+\
    '_3mm.nii.gz -ref /media/DATA/fmri/MNI152_T1_3mm -applyisoxfm 3 -interp nearestneighbour'
    print command
        
        
########### Atlas AAL conversion #################

atlas_list = os.listdir('/media/robbis/DATA/fmri/templates_AAL/')
atlas_list = [a for a in atlas_list if a.find('.img') != -1]

brain_mask = '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain_mask.nii.gz'
mni_img = '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm.nii.gz'

for a in atlas_list:
    
    input_img = os.path.join('/media/robbis/DATA/fmri/templates_AAL/',a)
    fout = a.split('.')[0]+'_mni'
    output_img = os.path.join('/media/robbis/DATA/fmri/templates_AAL/',fout)
    command = 'flirt -in '+input_img+' -out '+output_img+' -ref '+mni_img+' -applyisoxfm 1'
    print command
    
    command = 'fslmaths '+output_img+' -mul  '+brain_mask+' '+output_img
    print command

###############  Preprocessing of fcMRI  ###########################

mm = '2mm'
ref_img = '/usr/share/data/fsl-mni152-templates/MNI152_T1_%s.nii.gz' % (mm)

####### Registration to MNI ########
for d in dlist[:1]:

    mprage = os.path.join(path, d, 'mprage', 'mprage_orient.nii.gz')
    
    omat_fname = 'anat2mni_%s.mat' % (mm)
    omat = os.path.join(path, d, 'mprage', omat_fname)
    
    out_img_fname = 'mprage_orient_mni_%s.nii.gz' % (mm)
    out_img =  os.path.join(path, d, 'mprage', out_img_fname)
    
    command = 'flirt -in '+mprage+' '+\
              '-ref '+ref_img+' '+\
              '-omat '+omat+' '+\
              '-out '+out_img
    
    print command


###### Atlas registration to ANAT #######

atlas_path = '/media/robbis/DATA/fmri/templates_fcmri/0_findlab/'
pattern = 'separated'

atlas_list = os.listdir(atlas_path)
atlas_list = [a for a in atlas_list if a.find(pattern) != -1]


for d in dlist[:1]:

    omat_fname = 'anat2mni_%s.mat' % (mm)
    omat = os.path.join(path, d, 'mprage', omat_fname)
    
    imat_fname = 'inv_anat2mni_%s.mat' % (mm)
    imat = os.path.join(path, d, 'mprage', imat_fname)
    
    ref = os.path.join(path, d, 'mprage', 'mprage_orient.nii.gz')
    
    command = 'convert_xfm -omat '+imat+' -inverse '+omat

    print command 
    
    for a in atlas_list:
        input_img = os.path.join(atlas_path, a)
        output_img = os.path.join(path, d, 'mprage',a.split('.')[0]+'_anat')
        
        command = 'flirt -in '+input_img+' -ref '+ref+' -applyxfm -init '+imat+' '+\
                ' -out '+output_img+' -interp nearestneighbour'
                
        print command

####### Brain Extraction Tool ##############   
for d in dlist:
    mprage = os.path.join(path, d, 'fmri', 'bold_orient_mean')
    output = os.path.join(path, d, 'fmri', 'bold_orient_mask')
    
    command = 'bet2 '+mprage+' '+output+' -m -n'
    
    print command

######## Segmentation #######
for d in dlist:
    
    mprage = os.path.join(path, d, 'mprage', 'mprage_orient_brain.nii.gz')
    command = 'fast -g '+mprage
    print command

#### Rename segmentation files ########
for d in dlist:
    seg_list = os.listdir(os.path.join(path, d, 'mprage'))
    seg_list = [s for s in seg_list if s.find('_0.') != -1 
                or s.find('_1.') != -1 or s.find('_2.') != -1]
    
    for t in seg_list:
        tissue = os.path.join(path, d, 'mprage', t)
        if t.find('0') != -1:
            suffix = 'csf.nii.gz'
        elif t.find('1') != -1:
            suffix = 'gm.nii.gz'
        else:
            suffix = 'wm.nii.gz'
        
        nt = t.split('_')[:-1]
        nt.append(suffix)
        
        outname = '_'.join(nt)
        outname = os.path.join(path, d, 'mprage', outname)
        command = 'mv '+tissue+' '+outname
        
        print command

##### Gray Matter Intersection with Atlas ######

atlas_pattern1 = 'separated'
atlas_pattern2 = 'anat.nii'
script_filename = 'gm_findlab_intersect.sh'

'''
atlas_pattern1 = 'atlas'
atlas_pattern2 = 'mni_anat'
script_filename = 'gm_aal_intersect.sh'
'''
for d in dlist[:]:
    fname = open(os.path.join(path, d, script_filename), 'w')
    flist = os.listdir(os.path.join(path, d, 'mprage'))
    tissue_list = [t for t in flist if t.find('gm') != -1 and t.find('atlas') == -1 
                   and t.find('brain') != -1]
    atlas_list = [t for t in flist if t.find(atlas_pattern1) != -1 and t.find(atlas_pattern2) != -1]
    
    
    for t in tissue_list:
        tissue_fname = os.path.join(path, d, 'mprage',t)
        suffix_ = t.split('_')[2:]
        for a in atlas_list:
            atlas_fname = os.path.join(path, d, 'mprage',a)
            prefix_ = a.split('_')[:1]
            outname = '_'.join(prefix_+suffix_)
            out_fname = os.path.join(path, d, 'mprage',outname)
            command = 'fslmaths '+tissue_fname+' -mul '+atlas_fname+' '+out_fname
            fname.write(command+'\n')
            #print command
    fname.close()
    print 'sh '+os.path.join(path, d, script_filename)

###### Anatomical registration to fMRI ##############

for d in dlist:
    bold = os.path.join(path, d, 'fmri', 'bold_orient.nii.gz')
    bold_mean = os.path.join(path, d, 'fmri', 'bold_orient_mean.nii.gz')
    
    # Mean bold image
    command = 'fslmaths '+bold+' -Tmean '+bold_mean
    print command
    
    mprage = os.path.join(path, d, 'mprage', 'mprage_orient_brain.nii.gz')
    outmat = os.path.join(path, d, 'fmri', 'anat2fmri.mat')
    outimg = os.path.join(path, d, 'mprage', 'mprage_orient_brain_333.nii.gz')
    
    #Registration of anatomical
    command = 'flirt -in '+mprage+' -ref '+bold_mean+' -out '+outimg+' -omat '+outmat
    print command
    
######## Segmented images registration to fMRI space #########
for d in dlist[:]:
    script_name = open(os.path.join(path, d, 'atlas2fmri.sh'), 'w')
    atlas_list = os.listdir(os.path.join(path, d, 'mprage'))
    ref = os.path.join(path, d, 'fmri', 'bold_orient_mean.nii.gz')
    aal_list = [a for a in atlas_list if a.find('brain')!= -1 and a.find('atlas')!=-1]

    tis_list = [a for a in atlas_list if (a.find('brain.')== -1 and 
                                        a.find('brain_333')==-1) and 
                                        a.find('orient_brain')!=-1]
    
    imat = os.path.join(path, d, 'fmri', 'anat2fmri.mat')
    for a in aal_list+tis_list: 
        input_img = os.path.join(path, d, 'mprage', a)
        output_img = a.split('.')[0]+'_333.nii.gz'
        output_img = os.path.join(path, d, 'fmri', output_img)
        command = 'flirt -in '+input_img+' -ref '+ref+' -applyxfm -init '+imat+' '+\
                ' -out '+output_img+' -interp nearestneighbour'
        #print command
        script_name.write(command)
        script_name.write('\necho '+command)
        script_name.write('\n')
    
    script_name.close()
    command = 'sh '+os.path.join(path, d, 'atlas2fmri.sh')
    
    print command

########################################################################
##### Iso voxel for non iso voxel images ########
script_name = open(os.path.join(path, 'isovoxelize.sh'), 'w')
for d in dlist[-4:]:
    flist = os.listdir(os.path.join(path, d, 'mprage'))
    flist.remove('mprage_orient.nii.gz')
    flist.remove('mprage_orient_111.nii.gz')   
    flist.remove('mprage_orient_mni.nii.gz')
    flist = [f for f in flist if f.find('.nii.gz') != -1]
    
    ref_img = os.path.join(path, d, 'mprage', 'mprage_orient_111.nii.gz')
    new_dir = os.path.join(path, d, 'mprage', 'no_iso_voxel')
    command = 'mkdir '+new_dir
    script_name.write(command)
    script_name.write('\n')
    for f in flist:
        input_img = os.path.join(path, d, 'mprage', f)
        output_img = f.split('.')[0]+'_111.nii.gz'
        output_img = os.path.join(path, d, 'mprage', output_img)
        command = 'flirt -in '+input_img+\
                       ' -out '+output_img+\
                       ' -ref '+ref_img+\
                       ' -applyisoxfm 1'+\
                       ' -interp nearestneighbour'
        print command
        script_name.write(command)
        script_name.write('\n')
        
        command = 'mv '+input_img+' '+new_dir
        print command
        script_name.write(command)
        script_name.write('\n')
        
        command = 'mv '+output_img+' '+input_img
        print command
        script_name.write(command)
        script_name.write('\n')
        
script_name.close()
    

for d in dlist:
    input_img = os.path.join(path, d, 'mprage', 'mprage_orient_brain.nii.gz')
    command = 'rm '+input_img
    print command
    
    real_img = os.path.join(path, d, 'mprage', 'mprage_orient_brain_g.nii.gz')
    command = 'mv '+real_img+' '+input_img
    print command

atlas_networks = ['Auditory', 'Basal', 'LECN', 'Language', 'Precuneus', 'RECN',
       'Sensorimotor', 'Visuospatial', 'anterior', 'dorsal', 'high', 'post', 'prim',
       'ventral']

for d in dlist:
    atlas_img = os.listdir(os.path.join(path, d, 'mprage'))
    for a in atlas_networks:
        for p in ['pve', 'seg']:
            rm_image = ("%s_%s_gm.nii.gz") % (a, p)
            command = 'rm '+os.path.join(path, d, 'mprage', rm_image)
            print command


############# 4dfp ##########################
path_4dfp = '/media/robbis/DATA/fmri/template_4dfp/4dfp_refdir'
list_files = os.listdir(path_4dfp)
list_files = [f for f in list_files if f.find('4dfp.ifh') != -1]
for f in list_files:
    f_ = f.split('.ifh')[0]
    print 'nifti_4dfp -n '+os.path.join(path_4dfp, f_)+' '+os.path.join(path_4dfp, f_)
    print 'mv '+os.path.join(path_4dfp, f_)+'.nii '+os.path.join(path_4dfp, 'nifti')
    
    
    
####### Bold to MNI registration ####################
path_templates = '/media/robbis/DATA/fmri/templates_MNI_3mm/' 
for s in subjects:
    
    sub_dir = os.path.join(path, s)
    image = os.path.join(path, s, 'mprage', 'mprage_orient_brain.nii.gz')
    
    
    ## Register anat to mni
    command = 'flirt -in '+ image + \
                    ' -ref '+os.path.join(path_templates, 'MNI152_T1_3mm_brain.nii.gz') + \
                    ' -omat '+os.path.join(sub_dir, 'mprage','mprage2mni_3mm.mat') + \
                    ' -o '+ os.path.join(sub_dir, 'mprage','mprage2mni_3mm.nii.gz')
                    

    print command
    
    ## Register anat to bold
    command = 'flirt -in '+ image + \
                    ' -ref '+os.path.join(sub_dir, 'fmri', 'bold_orient_mean.nii.gz') + \
                    ' -omat '+os.path.join(sub_dir, 'fmri','mprage2bold_3mm.mat') + \
                    ' -o '+ os.path.join(sub_dir, 'mprage','mprage2bold_3mm.nii.gz')
                    
    print command
    
    ## Invert anat2bold matrix
    command = 'convert_xfm -omat %s -inverse %s' % (os.path.join(sub_dir, 'fmri','bold2mprage_3mm.mat'),
                                                    os.path.join(sub_dir, 'fmri','mprage2bold_3mm.mat'))
    print command
    
    ## Concatenate bold2anat2mni matrices
    command = 'convert_xfm -omat %s -concat %s %s' %(os.path.join(sub_dir, 'fmri','bold2mni_3mm.mat'),
                                                     os.path.join(sub_dir, 'mprage','mprage2mni_3mm.mat'),
                                                     os.path.join(sub_dir, 'fmri','bold2mprage_3mm.mat'))
    print command
    
    ## Transform bold2mni             
    command = 'applyxfm4D '+ os.path.join(sub_dir, 'fmri', 'bold_orient.nii.gz') + ' ' \
                           + os.path.join(path_templates, 'MNI152_T1_3mm_brain.nii.gz') + ' ' \
                           + os.path.join(sub_dir, 'fmri', 'bold_orient_mni_3mm.nii.gz') + ' ' \
                           +  os.path.join(sub_dir, 'fmri','bold2mni_3mm.mat') + ' ' \
                           + '-singlematrix'   
    print command          

filelist = []
for s in subjects:
    sub_dir = os.path.join(path, s)
    filelist.append(os.path.join(sub_dir, 'fmri', 'bold_orient_mni_3mm.nii.gz'))
command = 'slicesdir -p' +os.path.join(path_templates, 'MNI152_T1_3mm.nii.gz') + ' ' +' '.join(filelist)

print command


path_results = '/media/robbis/DATA/fmri/monks/0_results/20131201_073007_searchlight_total_fmri/'

for s in subjects:

    image = os.path.join(path_results, s, s+"_radius_3_searchlight_mean_map.nii.gz")        

    command = 'fslswapdim %s -z -x -y %s ' %(image, image[:-7]+"_orient.nii.gz")
    print command
    
    command = 'fslorient -setqformcode 1 %s ' %(image[:-7]+"_orient.nii.gz")
    print command
    
    
for s in subjects:
    sub_dir = os.path.join(path, s)
    command = 'applyxfm4D '+ os.path.join(path_results, s, s+"_radius_3_searchlight_mean_map_orient.nii.gz") + ' ' \
                           + os.path.join(path_templates, 'MNI152_T1_3mm_brain.nii.gz') + ' ' \
                           + os.path.join(path_results, s, s+"_radius_3_searchlight_mean_map_mni_3mm.nii.gz") + ' ' \
                           +  os.path.join(sub_dir, 'fmri','bold2mni_3mm.mat') + ' ' \
                           + '-singlematrix'   
    print command 