#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

# pylint: disable=maybe-no-member, method-hidden

import os
import nibabel as ni


def roi_wu_data(path, name, task, init_vol=0):
    
    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('.img') != -1) & (elem.find('.rec') == -1)]
    
    for img in imgFiles:
        
        n_vols = ni.analyze.load(os.path.join(path, name, task, img)).get_shape()[3]
        
        roi = 'fslroi '+ \
            os.path.join(path, name, task, img) + ' ' + \
            os.path.join(path, name, task, img[:-4] + '_crop') + ' ' +\
            str(init_vol) + ' ' + str(n_vols - init_vol)
        print roi
        os.system(roi)
    
    
def bet_wu_data_(path, name, task):
    
    task = ''
    print '--------- BRAIN EXTRACTION ----------'

    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    #imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('crop.') != -1) & (elem.find('.rec') == -1)]
    imgFiles = [elem for elem in imgFiles if (elem.find('hdr') != -1)]
   
    for img in imgFiles:
        
        bet = 'bet '+ \
            os.path.join(path, name, task, img) + ' ' + \
            os.path.join(path, name, task, img[:-4] + '_brain') + ' ' +\
            '-F -f 0.60 -g 0'
        print bet
        os.system(bet)


def mc_wu_data(path, name, task):
     
    # nameL = os.listdir('/media/DATA/fmri/learning/')
    # nameL = [elem for elem in nameL if (elem.find('_') == -1) & (elem.find('.') == -1)]
    print '      ---- > Motion Correction <-----   '

    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('brain.') != -1) & (elem.find('.rec') == -1)]
    
    imgFiles.sort()
    
    restList = os.listdir(path+'/'+name+'/rest')
    restList = [elem for elem in restList if (elem.find('.img_') == -1) & (elem.find('rest1') != -1) & (elem.find('brain.') != -1) \
                                                                    &   (elem.find('.rec') == -1)]
    ref = restList.pop()
    
    i = 0
    for img in imgFiles:
        i = i + 1
        
        imgIn = path+'/'+name+'/'+task+'/'+img
        refIn = path+'/'+name+'/rest/'+ref

        command =   'flirt '+ \
                    ' -in ' + imgIn + \
                    ' -ref '+refIn+ \
                    ' -searchcost normmi' +\
                    ' -omat '+path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.mat'+ \
                    ' -dof 12'

        print command
        os.system(command)
       
    i = 0
    for img in imgFiles:  
           
        i = i + 1
        imgIn = path+'/'+name+'/'+task+'/'+img
        refIn = path+'/'+name+'/rest/'+ref

        command = 'flirt '+ \
                  ' -in '+imgIn+ \
                  ' -ref '+refIn+ \
                  ' -init '+path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.mat'+ \
                  ' -applyxfm -interp nearestneighbour' + \
                  ' -out '+ path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.nii.gz' 
                  
        print command         
        os.system(command)

def wu_to_mni (path, name, task): 
    
    print '      ---- > MNI Coregistration <-----   '
    
    imgFiles = os.listdir( os.path.join(path,name,task) )
    imgFiles = [elem for elem in imgFiles if (elem.find('flirt.nii.gz') != -1) and (elem.find('mni') == -1)]
    
    imgFiles.sort()
   
    #ref = imgFiles[len(imgFiles)-1]
    
    i=0
    
    for img in imgFiles:
        i = i + 1
        imgIn = os.path.join(path,name,task,img)
        refIn = '/media/DATA/fmri/MNI152_T1_3mm_brain.nii.gz'

        command =   'flirt '+ \
                    ' -in ' + imgIn + \
                    ' -ref '+refIn+ \
                    ' -omat '+  os.path.join(path,name,task,img[:-7]) +'_mni.mat'+ \
                    ' -dof 12' + \
                    ' -searchcost normmi' + \
                    ' -searchrx -90 90' + \
                    ' -searchry -90 90' + \
                    ' -searchrz -90 90'
        
        print command
        os.system(command)
       
    i = 0
    for img in imgFiles:  
           
        i = i + 1
        imgIn = os.path.join(path,name,task,img)

        command = 'flirt '+ \
                  ' -in '+imgIn+ \
                  ' -ref '+refIn+ \
                  ' -init '+os.path.join(path,name,task,img[:-7]) +'_mni.mat'+ \
                  ' -applyxfm' + \
                  ' -out '+ os.path.join(path,name,task,img[:-7]) +'_mni.nii.gz' 
                  
        print command         
        os.system(command)
        
########### Monks functions ######################        

def mni_registration(path, subjects, template_mm='2mm', **kwargs):
    
    mm = template_mm
    ref_img = '/usr/share/data/fsl-mni152-templates/MNI152_T1_%s.nii.gz' % (mm)
    
    ####### Registration to MNI ########
    output_file = 'anat2mni_%s.sh' % (mm)
    output_path = os.path.join(path, output_file)
    
    file_ = open(output_path, 'w')
    
    for d in subjects:
    
        mprage = os.path.join(path, d, 'mprage', 'mprage_orient.nii.gz')
        
        omat_fname = 'anat2mni_%s.mat' % (mm)
        omat = os.path.join(path, d, 'mprage', omat_fname)
        
        out_img_fname = 'mprage_orient_mni_%s.nii.gz' % (mm)
        out_img =  os.path.join(path, d, 'mprage', out_img_fname)
        
        command = 'flirt -in '+mprage+' '+\
                  '-ref '+ref_img+' '+\
                  '-omat '+omat+' '+\
                  '-out '+out_img
        
        file_.write(command)
        file_.write('\n')
        #print command
    
    file_.close()   
    print 'sh '+output_path



def atlas_anatomical_registration(path, subjects, atlas_path, pattern, template_mm='2mm'):
    
    mm = template_mm
    #atlas_path = '/media/robbis/DATA/fmri/templates_fcmri/0_findlab/'
    #pattern = 'separated'
    
    atlas_list = os.listdir(atlas_path)
    atlas_list = [a for a in atlas_list if a.find(pattern) != -1]
    
    out_fname = 'atlas_to_anat_%s.sh' % (mm)
    out_script = os.path.join(path, out_fname)
    
    
    file_ = open(out_script, 'w')
    
    for d in subjects:
    
        omat_fname = 'anat2mni_%s.mat' % (mm)
        omat = os.path.join(path, d, 'mprage', omat_fname)
        
        imat_fname = 'inv_anat2mni_%s.mat' % (mm)
        imat = os.path.join(path, d, 'mprage', imat_fname)
        
        ref = os.path.join(path, d, 'mprage', 'mprage_orient.nii.gz')
        
        command = 'convert_xfm -omat '+imat+' -inverse '+omat
        
        file_.write(command)
        file_.write('\n')
        
        #print command 
        
        for a in atlas_list:
            input_img = os.path.join(atlas_path, a)
            output_img = os.path.join(path, d, 'mprage',a.split('.')[0]+'_anat')
            
            command = 'flirt -in '+input_img+' -ref '+ref+' -applyxfm -init '+imat+' '+\
                    ' -out '+output_img+' -interp nearestneighbour'
                    
            file_.write(command)
            file_.write('\n')        
                    
            #print command
    print 'sh '+out_script
    file_.close()
    
    
def gray_matter_intersection(path, subjects, atlas='findlab'):
    
    
    if atlas == 'findlab':
        atlas_pattern1 = 'separated'
        atlas_pattern2 = 'anat.nii'
        script_filename = 'gm_findlab_intersect.sh'
    else:
        atlas_pattern1 = 'atlas'
        atlas_pattern2 = 'mni_anat'
        script_filename = 'gm_aal_intersect.sh'


    for d in subjects:
        fname = open(os.path.join(path, d, script_filename), 'w')
        flist = os.listdir(os.path.join(path, d, 'mprage'))
        # 
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


def atlas_bold_registration(path, subjects, atlas='findlab'):
    
    if atlas == 'findlab':
        pattern1 = 'brain'
        pattern2 = 'gm'
    else:
        pattern1 = 'brain'
        pattern2 = 'atlas'
        
    script_fname = 'atlas2fmri_%s.sh' % (atlas)
    for d in subjects:
        script_name = open(os.path.join(path, d, script_fname), 'w')
        atlas_list = os.listdir(os.path.join(path, d, 'mprage'))
        ref = os.path.join(path, d, 'fmri', 'bold_orient_mean.nii.gz')
        aal_list = [a for a in atlas_list if a.find(pattern1)!= -1 and a.find(pattern2)!=-1]
    
        tis_list = [a for a in atlas_list if (a.find('brain.')== -1 and 
                                            a.find('brain_333')==-1) and 
                                            a.find('orient_brain')!=-1]
        
        imat = os.path.join(path, d, 'fmri', 'anat2fmri.mat')
        for a in aal_list:#+tis_list: 
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
        command = 'sh '+os.path.join(path, d, script_fname)
        
        print command