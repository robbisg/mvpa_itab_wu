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
    