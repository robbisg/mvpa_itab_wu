import os
import nibabel as ni
import numpy as np
import matplotlib.pyplot as plt
import logging
import cPickle as pickle

from ..plot import plot_clusters_graph
from ..results import get_time

logger = logging.getLogger(__name__)

def save_results(path, results, configuration):
    """
    path: is the results path dir
    results: is the structure used to store data
    configuration: is the configuration file to store analysis info
    """
    
    # Get information to make the results dir
    datetime = get_time()
    analysis = configuration['analysis_type']
    mask = configuration['mask_area']
    task = configuration['analysis_task']
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    # New directory to store files
    parent_dir = os.path.join(path, '0_results', new_dir)
    
    # Check which analysis has been done
    ## TODO: Use a class to save results
    if analysis == 'searchlight':
        save_results_searchlight(parent_dir, results)
    elif analysis == 'transfer_learning':
        save_results_transfer_learning(parent_dir, results)
        #write_all_subjects_map(path, new_dir)
    elif analysis == 'clustering':
        save_results_clustering(parent_dir, results)
    else:
        save_results_basic(parent_dir, results)
        write_all_subjects_map(path, new_dir)  

    ###################################################################        
    import csv
    w = csv.writer(open(os.path.join(parent_dir, 'configuration.csv'), "w"))
    for key, val in configuration.items():
        w.writerow([key, val])
        
    #######################################################################
    
    if analysis == 'searchlight':
        logger.info('Result saved in '+parent_dir)
        return 'OK' #WTF?
    else:
        file_summary = open(os.path.join(parent_dir, 'analysis_summary_'+mask+'_'+task+'.txt'), "w")
        res_list = []
    
        for subj in results.keys():
        
            list = [subj]
            file_summary.write(subj+'\t')
            
            r = results[subj]
        
            list.append(r['stats'].stats['ACC'])
            list.append(r['p'])
            
            file_summary.write(str(r['stats'].stats['ACC'])+'\t')
            file_summary.write(str(r['p'])+'\t')
            
            if analysis == 'transfer_learning':
                list.append(r['confusion_total'].stats['ACC'])
                list.append(r['d_prime'])
                list.append(r['beta'])
                list.append(r['c'])
                
                file_summary.write(str(r['confusion_total'].stats['ACC'])+'\t')
                file_summary.write(str(r['d_prime'])+'\t')
                file_summary.write(str(r['beta'])+'\t')
                file_summary.write(str(r['c'])+'\t')
                
            file_summary.write('\n')
            
        res_list.append(list)
        
        file_summary.close()
    
    logger.info('Result saved in '+parent_dir)
    
    return 'OK' 

def save_results_searchlight (path, results):
    
    parent_dir = path 
    
    total_map = []
    
    # For each subject save map and average across fold
    for name in results:
        
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        map_ = results[name]['map']
        
        radius = np.int(results[name]['radius'])
        
        
        if len(map_.get_data().shape) > 3:
            mean_map = map_.get_data().mean(axis=3)
            mean_img = ni.Nifti1Image(mean_map, affine=map_.get_affine())
            fname = name+'_radius_'+str(radius)+'_searchlight_mean_map.nii.gz'
            ni.save(mean_img, os.path.join(results_dir,fname))
        else:
            mean_map = map_.get_data()
        
        
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'
        ni.save(map_, os.path.join(results_dir,fname))
        
        total_map.append(mean_map)
    
    # Save the total average map
    total_map = np.array(total_map).mean(axis=0)
    total_img = ni.Nifti1Image(total_map, affine=map_.get_affine())
    fname = 'accuracy_map_radius_'+str(radius)+'_searchlight_all_subj.nii.gz'
    ni.save(total_img, os.path.join(path,fname))
                   
    logger.info('Results writed in '+path)
    return path

def save_results_basic(path, results):
    
    parent_dir = path 
    
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        for key in results[name]:
            
            if key == 'map':
                
                m_mean = results[name]['map'].pop()
                m_mean_data = m_mean.get_data()
                fname = name+'_mean_map.nii.gz'
                m_mean_data = (m_mean_data - np.mean(m_mean_data))/np.std(m_mean_data)
                
                m_mean_zscore = ni.Nifti1Image(m_mean_data, m_mean.get_affine())
                ni.save(m_mean_zscore, os.path.join(results_dir,fname))
                
                #save_map(os.path.join(results_dir, fname), m_mean_data, m_mean.get_affine())
                
                for map, t in zip(results[name][key], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
                    map_data = map.get_data()
                    map_data_zscore = (map_data - np.mean(map_data))/np.std(map_data)
                    map_zscore = ni.Nifti1Image(map_data_zscore, map.get_affine())
                    ni.save(map_zscore, os.path.join(results_dir,fname))
                    
            elif key == 'stats':
                stats = results[name][key]
                fname = name+'_stats.txt'
                file = open(os.path.join(results_dir,fname), 'w')
                file.write(str(stats))
                file.close()
            else:
                obj = results[name][key]
                if key == 'classifier':
                    obj = results[name][key].ca
                fname = name+'_'+key+'.pyobj'          
                file = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file)
                file.close()
     
    ###################################################################          
    
    
    logger.info('Result saved in '+parent_dir)
    
    return 'OK' 


def save_results_transfer_learning(path, results):
    
    # Cross-decoding predictions and labels
    # p = classifier prediction on target ds
    # r = targets of target ds
    p = results[results.keys()[0]]['mahalanobis_similarity'][0].T[1]
    r = results[results.keys()[0]]['mahalanobis_similarity'][0].T[0]
    
    # Stuff for total histograms 
    hist_sum = dict()
    
    means_s = dict()
    for t in np.unique(r):
        hist_sum[t] = dict()
        for l in np.unique(p):
            hist_sum[t][l] = dict()
            hist_sum[t][l]['dist'] = []
            hist_sum[t][l]['p'] = []
            means_s[l+'_'+t] = []
    
    
    for name in results:
        # Make subject dir
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name)                        
        
        # Statistics of decoding
        stats = results[name]['stats']
        fname = name+'_stats.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        file_.write(str(stats))
        
        p_value = results[name]['pvalue']
        file_.write('\n\n p-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        
        file_.write('\n\n Mean each fold p-value: '+str(p_value.mean()))
        file_.write('\n\n Mean null dist total accuracy value: '+str(results[name]['p']))
        file_.write('\nd-prime coefficient: '+str(results[name]['d_prime']))
        file_.write('\nbeta coefficient: '+str(results[name]['beta']))
        file_.write('\nc coefficient: '+str(results[name]['lab']))
        #file.write('\n\nd-prime mahalanobis coeff: '+str(results[name]['d_prime_maha']))
        file_.close()
        
        if name == 'group':
            fname = name+'_fold_stats.txt'
            file_ = open(os.path.join(results_dir,fname), 'w')
            for m in stats.matrices:
                file_.write(str(m.stats['ACC']))
                file_.write('\n')
            file_.close()
        
        
        for k in results[name].keys():
            if k in ['classifier', 'targets', 'predictions']:
                if k == 'classifier':
                    obj = results[name][k].ca
                else:
                    obj = results[name][k]
                    
                fname = name+'_'+k+'.pyobj'
        
                file_ = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file_)
                file_.close()
        
            if k in ['confusion_target', 'confusion_total']:
                c_m = results[name][k]
                fname = name+'_'+k+'.txt'
                file_ = open(os.path.join(results_dir,fname), 'w')
                file_.write(str(c_m))
                file_.close()
                
                
        #plot_transfer_graph(results_dir, name, results[name])      
        
        ####################### Similarity results #####################################
        # TODO: Keep in mind! They should be saved when similarity has been performed!!!
        full_data = results[name]['mahalanobis_similarity'][0]
        similarity_mask = results[name]['mahalanobis_similarity'][1]
        threshold = results[name]['mahalanobis_similarity'][2]
        distances = results[name]['mahalanobis_similarity'][4]
        
        # Renaming variables to better read
        ds_targets = full_data.T[0]
        class_prediction_tar = full_data.T[1]
        
        t_mahala = full_data[similarity_mask]
        fname = name+'_mahalanobis_data.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        n_src_label = len(np.unique(class_prediction_tar))
        n_tar_label = len(np.unique(ds_targets))
        
        plot_list = dict()
        
        # For each label in predictions
        for t in np.unique(class_prediction_tar):
            f, ax = plt.subplots(2, 1)
            plot_list[t] = [f, ax]
        

        
        
        # For each label of the target dataset (target classes)
        for tar in np.unique(ds_targets): 
            
            # Select data belonging to target loop class
            target_mask = ds_targets == tar
            similarity_target = similarity_mask[target_mask]
            target_data = full_data[target_mask] 


            for lab in np.unique(class_prediction_tar):
 
                # Select target data classified as loop label
                prediction_mask = target_data.T[1] == lab 
                crossd_data = target_data[prediction_mask] 
                similarity_crossd = similarity_target[prediction_mask]

                distance_ = crossd_data.T[2]
                p_values_ = crossd_data.T[3]                
                
                # Filter data that meets similarity criterion
                similarity_data = crossd_data[similarity_crossd]
                num = len(similarity_data)
                
                
                if len(target_data) != 0:
                    distance_data = similarity_data.T[2]# Mean distance
                    p_data = similarity_data.T[3] # Mean p-value
                else:
                    distance_data = np.mean(np.float_(distance_)) 
                    p_data = np.mean(np.float_(p_values_)) # 
                
                mean_maha_d = np.mean(np.float_(distance_data))
                mean_maha_p = np.mean(np.float_(p_data))
                
                tot_d = np.mean(np.float_(distance_))
                tot_p = np.mean(np.float_(p_values_))
                
                occurence_ = ','.join([tar, lab, str(num), str(mean_maha_d),str(mean_maha_p),str(tot_d),str(tot_p)])
                
                file_.write(occurence_+'\n')
                
                # TODO: Maybe is possible to collapse both file in a single one!
                histo_d_fname = "%s_hist_%s_%s_dist.txt" % (name, lab, tar)
                histo_p_fname = "%s_hist_%s_%s_p.txt" % (name, lab, tar)
                
                np.savetxt(os.path.join(results_dir,histo_d_fname), np.float_(distance_))
                np.savetxt(os.path.join(results_dir,histo_p_fname), np.float_(p_values_))
                
                # Histogram plots
                # TODO: Maybe it's better to do something else!
                # TODO: Unique values of bins!
                plot_list[lab][1][0].hist(np.float_(distance_), bins=35, label=tar, alpha=0.5)
                plot_list[lab][1][1].hist(np.float_(p_values_), bins=35, label=tar, alpha=0.5)          
                
                # We store information for the total histogram
                hist_sum[tar][lab]['dist'].append(np.float_(distance_))
                hist_sum[tar][lab]['p'].append(np.float_(p_values_))
                
                
                
                ## TODO: Insert plot in a function and let the user decide if he wants it!
                ## plot_distances(distances, runs)
                ## distance_ = data[prediction_mask] # Equivalent form!
                data = distances[lab][target_mask]
                f_d = plt.figure()
                a_d = f_d.add_subplot(111)
                a_d.plot(data)
                a_d.set_ylim(data.mean()-3*data.std(), data.mean()+3*data.std())
                step = data.__len__() / 6.
                for j in np.arange(6)+1:#n_runs
                    a_d.axvline(x = step * j, ymax=a_d.get_ylim()[1], color='y', linestyle='-', linewidth=1)
                a_d.axhline(y = threshold, color='r', linestyle='--', linewidth=2)
                
                
                a_d.axhline(y = np.mean(data), color='black', linestyle=':', linewidth=2)
                
                pname = "%s_distance_plot_%s_%s.png" % (name, lab, tar)
                
                f_d.savefig(os.path.join(results_dir, pname))
                
                ## Save file ##
                fname = "%s_distance_txt_%s_%s.txt" % (name, lab, tar)
                np.savetxt(os.path.join(results_dir, fname), 
                           data, fmt='%.4f')               

                means_s[lab+'_'+tar].append(np.mean(data))
                
        ## TODO: Insert in a function        
        for k in plot_list.keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            fig_name = os.path.join(results_dir,name+'_histogram_'+k+'.png')
            plot_list[k][0].savefig(fig_name)

        file_.write('\nthreshold '+str(threshold))       
        file_.close()
        
        cmatrix_mahala = results[name]['confusion_mahala']
        fname = "%s_confusion_mahala.txt" % name
        file_ = open(os.path.join(results_dir,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            print err
        
        file_.close()
        
        # Is this snippert a part of other code???
        if results[name]['map'] != None:
            
            m_mean = results[name]['map'].pop()
            fname = name+'_mean_map.nii.gz'
            mask_ = m_mean._data != 0
            m_mean._data[mask_ ] = (m_mean._data[mask_ ] - np.mean(m_mean._data[mask_ ]))/np.std(m_mean._data[mask_ ])
            ni.save(m_mean, os.path.join(results_dir,fname))
        
            for map, t in zip(results[name]['map'], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
    
                    #map._data = (map._data - np.mean(map._data))/np.std(map._data)
                    map._dataobj = (map._dataobj - np.mean(map._dataobj))/np.std(map._dataobj)
                    ni.save(map, os.path.join(results_dir,fname))
    
    
    '''
    End of main for
    '''
                
    from pprint import pprint
    filename_ = open(os.path.join(path, 'distance_means.txt'), 'w')
    pprint(means_s, filename_)
    filename_.close()
    
    plot_list = dict()
    for t in hist_sum[hist_sum.keys()[0]].keys():
        f, ax = plt.subplots(2, 1)
        plot_list[t] = [f, ax]
    
    for k1 in hist_sum:
        
        for k2 in hist_sum[k1]:
            
            for measure in hist_sum[k1][k2]:
                hist_sum[k1][k2][measure] = np.hstack(hist_sum[k1][k2][measure])
                
        
            plot_list[k2][1][0].hist(hist_sum[k1][k2]['dist'], bins=35, label = k1, alpha=0.5)
            plot_list[k2][1][1].hist(hist_sum[k1][k2]['p'], bins=35, label = k1, alpha=0.5)
   
    for k in hist_sum[hist_sum.keys()[0]].keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            plot_list[k][0].savefig(os.path.join(path,'total_histogram_'+k+'.png'))
    
    plt.close('all')    
    return


def save_results_clustering(path, results):
    
    
    for name in results:
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name) 
        
        for key in results[name]:
            obj = results[name][key]
            fname = name+'_'+key+'.pyobj'          
            file = open(os.path.join(results_dir,fname), 'w')
            pickle.dump(obj, file)
            file.close()
            
            if key == 'clusters':
                plot_clusters_graph(results_dir, name, results[name])
  
    return

def save_map(filename, map_np_array, affine=np.eye(4)):
        
    map_zscore = ni.Nifti1Image(map_np_array, affine)
    ni.save(map_zscore, filename)
    return

def write_all_subjects_map(path, dir_):
    
    res_path = os.path.join(path, '0_results', dir_)
    
    subjects = os.listdir(res_path)
    subjects = [s for s in subjects if s.find('.') == -1]
    
    img_list = []
    i = 0
    list_t = []
    
    min_t = 1000
    max_t = 0
    
    #Get data from path
    for s in subjects:
        
        s_path = os.path.join(res_path, s)
        map_list = os.listdir(s_path)
        
        map_list = [m for m in map_list if m.find('mean_map') != -1]
        
        img = ni.load(os.path.join(s_path, map_list[0]))
        
        #Store maximum and minimum if maps are dimension mismatching
        
        if img.get_data().shape[-1] < min_t:
            min_t = img.get_data().shape[-1]
        
        if img.get_data().shape[-1] > max_t:
            max_t = img.get_data().shape[-1]
            
        img_list.append(img)
        list_t.append(img.get_data().shape[-1])
        i = i + 1
        
    dim_mask = np.array(list_t) == min_t
    img_list_ = np.array(img_list)
    
    #compensate for dimension mismatching
    n_list = []
    if dim_mask.all() == False:
        img_m = img_list_[dim_mask]
        n_list = []
        for img in img_m:
            data = img.get_data()
            for i in range(max_t - data.shape[-1]):
                add_img = np.mean(data, axis=len(data.shape)-1)
                add_img =  np.expand_dims(add_img, axis=len(data.shape)-1)
                data = np.concatenate((data, add_img), axis=len(data.shape)-1)
            n_list.append(data)   
        v1 = np.expand_dims(img_list_[~dim_mask][0].get_data(), axis=0)
    else:
        v1 = np.expand_dims(img_list_[dim_mask][0].get_data(), axis=0)
    
    for img in img_list_[~dim_mask][1:]:
        v1 = np.vstack((v1,np.expand_dims(img.get_data(), axis=0)))
    
    for img_data in n_list:
        v1 = np.vstack((v1,np.expand_dims(img_data, axis=0)))
        
    m_img = np.mean(v1, axis=0)
    
    img_ni = ni.Nifti1Image(m_img.squeeze(), img.get_affine())
    
    filename = os.path.join(res_path, 'all_subjects_mean_map_prova.nii.gz')
    ni.save(img_ni, filename)
    
    return
