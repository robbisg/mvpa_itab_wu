import seaborn as sns
import os
import nibabel as ni

path = '/media/robbis/DATA/fmri/memory/0_results/balanced_analysis/local/'
list_subj = os.listdir(path)
list_subj.sort()
colors = ['r', 'g', 'b']
list_subj = [s for s in list_subj if s.find('.') == -1 and s.find('_') == -1]

subj_collection = []

for s in list_subj:
    
    subj_imgs = get_images(path, s, 'decision')
    subj_collection.append(subj_imgs)
    pl.figure()
    for i, img in enumerate(subj_imgs):
        
        sns.distplot(img[img!=0].flatten(), 
                     color=colors[i], 
                     hist=False, 
                     kde_kws={"shade": True})

#############################

def get_images(path, subj, task_name):
    
    subject_path = os.path.join(path, subj)
    
    image_list = os.listdir(os.path.join(path, subj))
    image_list.sort()
    
    image_list = [i for i in image_list if i.find(task_name) != -1]
    image_list = [i for i in image_list if i.find('.nii.gz') != -1]
    image_list = [i for i in image_list if i.find('mean') != -1]
    
    img_container = []
    for img_name in image_list:
        img = ni.load(os.path.join(subject_path, img_name))
        print os.path.join(subject_path, img_name)
        img_container.append(img.get_data())
        
    
    return np.array(img_container)
        
