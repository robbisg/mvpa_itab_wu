import os
import nibabel as ni
import numpy as np
import time
import logging
from scipy.stats import ttest_1samp
import pickle




class Results(object):
    
    
    ## TODO: Could be useful do a summary header?
    
    def __init__(self, path, conf):
        
        self._result_array = []
        self.index = 0
        self.count = 0
        self.analysis_path = path
        self.summary = dict()
        
        # Get information to make the results dir
        datetime = get_time()
        self._mask = conf['mask_area']
        self._task = conf['analysis_task']
        self._analysis = conf['analysis_type']
        
        dir_ = '_'.join([datetime, self._analysis, self._mask, self._task])
        self.path = os.path.join(self.analysis_path, '0_results', dir_)
        
        make_dir(self.path)
        
    
    def add(self, result):
        # For each subject result added we make a new dir to store results
        if isinstance(result, SubjectResult):
            self._result_array.append(result)
            self.count+=1
            self._current_path = os.path.join(self.path, result.name)
            make_dir(self._current_path)
            self.summary[result.name] = []
            
            result.save(self._current_path)
    
    def summarize(self):
        
        fname = "analysis_summary_%s_%s.txt" %(self._mask, self._task)
        file_summary = open(os.path.join(self.path, fname), "w")
            
        for name, value in self.summary.items():
        
            file_summary.write(name+',')

            for v in value:
                file_summary.write(str(v)+',')
                
            file_summary.write('\n')
        
        file_summary.close() 
        

    
class ResultsCollection(Results):
    
    def __init__(self):
        self.collection = []        
    
    def summarize(self):
        super(ResultsCollection, self).summarize()
        for item in self.collection:
            item.summarize()   


        
class SearchlightResults(Results):
    
    expected_fields = ['name', 'map', 'radius']
    
    def __init__(self, path, conf):

        super(SearchlightResults, self).__init__(path, conf)
        self._total_map = []        


    def add(self, result):
        """
        When we add a result, we store new single subject results
        """
        super(SearchlightResults, self).add(result)
        
        if not isinstance(result, SearchlightSubjectResult):
            raise ValueError("Bad class added!")
        

        self._total_map.append(result._mean_map)

    
    def save(self):
        """
        Function who saves the union of subject's map to easily see it
        and the mean map across subjects
        """
        if self.count == 0:
            raise ValueError()        
        
        total_map = self._total_map
        affine = self._result_array[0]._map.get_affine()
        radius = self._result_array[0]._radius
            
        total_map = np.rollaxis(np.array(total_map), 0, 4)
        total_img = ni.Nifti1Image(total_map, affine=affine)
        
        fname = "accuracy_map_radius_%s_searchlight_all_subj.nii.gz" % radius
        ni.save(total_img, os.path.join(self.path,fname))
        
        mean_img = ni.Nifti1Image(total_map.mean(3), affine=affine)
        
        fname = "accuracy_map_radius_%s_searchlight_mean_subj.nii.gz" % radius
        ni.save(mean_img, os.path.join(self.path,fname))         
        
        logging.info('Results writed in '+self.path)
        
        self._total_map = total_map
        
        return self.path

        
    def summarize(self, threshold_value=0.5):
        """
        This is used to do basic tests
        """
        if self._total_map == []:
            raise ValueError("Try to do save() before summarize()!") 
        
        if self.count == 0:
            raise ValueError("Result object is empty!")
        
        affine = self._result_array[0]._map.get_affine()
        
        total_map = self._total_map
        
        t, p = ttest_1samp(total_map, threshold_value, axis=3)
        
        fname = "t_map_vs_threshold_%s_uncorrected.nii.gz" % threshold_value 
        _img = ni.Nifti1Image(t, affine=affine)
        ni.save(_img, os.path.join(self.path, fname))
        
        fname = "p_map_vs_threshold_%s_uncorrected.nii.gz" % threshold_value 
        _img = ni.Nifti1Image(p, affine=affine)
        ni.save(_img, os.path.join(self.path, fname))  



class DecodingResults(Results):
    
    def __init__(self, path, conf):
        super(DecodingResults, self).__init__(path,conf)
    
    
    def add(self, result):
        super(DecodingResults, self).add(result)
    
        self.summary[result.name].append(result._stats.stats['ACC'])
        self.summary[result.name].append(result._p)
    
    '''
    def summarize(self):
        # TODO: Write a mean map of weights??
    '''   



class SimilarityResults(Results):
        
    def __init__(self, path, conf):
        super(SimilarityResults, self).__init__(path,conf)
        self._histogram_data = []
        
        
    def add(self, result):
        super(SimilarityResults, self).add(result)
        
        self._histogram_data.append(result._data)
        self._unique_predictions = result._unique_predictions
        self._unique_targets = result._unique_targets
        self._items = result._values
        
    def summarize(self):
        
        import matplotlib.pyplot as pl
        
        for t in self._unique_targets:
            fig = pl.figure()
            
            for i, k_ in enumerate(self._items):
                
                ax = fig.add_subplot(i, 1, len(self._items))
                
                for p in self._unique_predictions:
                    
                    hist_data = []
                    
                    for data_ in self._histogram_data:
                        hist_data.append(data_[p][t][k_])
                    
                    hist_data = np.array(hist_data)
                
                    ax.hist(hist_data, bins=35, label=p, alpha=0.5)
        
        
                ax.legend()
            
            pname = os.path.join(self.path, "total_histogram_%s.png" % (t))
            fig.savefig(pname) 
    
    

class SignalDetectionResults(Results):
    
    def __init__(self, path, conf):
        Results.__init__(self, path, conf)
        
    def add(self, result):
        Results.add(self, result)
        
        self.summary[result.name].append(result._beta)
        self.summary[result.name].append(result._d_prime)
        self.summary[result.name].append(result._c)


class CrossDecodingResults(Results):
    
    def __init__(self, path, conf):
        Results.__init__(self, path, conf)
        
    def add(self, result):
        Results.add(self, result)
        
        self.summary[result.name].append(result._confusion_total.stats['ACC'])

  
        
class SubjectResult(object):
    
    def __init__(self, name, result_dict):
        self.name = name
        for k, v in result_dict.items():
            setattr(self, '_'+str(k), v)
    
    def add_result(self, res_name, value):
        setattr(self, res_name, value)


class SubjectResultCollection(SubjectResult):
    
    def __init__(self):
        self.collection = []
        
    def add(self, subj_result):
        self.collection.append(subj_result)
        
    def save(self, path):
        for result_ in self.collection:
            result_.save(path)

        

class CrossDecodingSubjectResult(SubjectResult):
    
    _fields = ['confusion_target', 'confusion_total']
    
    def __init__(self, name, result_dict):
        SubjectResult.__init__(self, name, result_dict)
        
    def save(self, path):
        
        for k in self._fields:
            matrix = getattr(self, '_'+k)
            fname = "%s_%s.txt" % (self.name, k)
            file_ = open(os.path.join(path, fname), 'w')
            file_.write(str(matrix))
            file_.close()


       

class SearchlightSubjectResult(SubjectResult):
            
    
    def save(self, path):
        """
        When we add a result we store new single subject results
        """
        
        name_ = self.name
        radius_ = np.int(self._radius)
        map_ = self._map
        
        print map_.get_affine()
        
        # If map is obtained via folding we average across maps
        if len(map_.get_data().shape) > 3: 
            mean_map = map_.get_data().mean(axis=3)
            mean_img = ni.Nifti1Image(mean_map, affine=map_.get_affine())
            fname = "%s_radius_%s_searchlight_mean_map.nii.gz" % (name_, str(radius_))    
            ni.save(mean_img, os.path.join(path,fname))
            return_map = mean_map
        else:
            return_map = map_.get_data()
                
        fname = "%s_radius_%s_searchlight_total_map.nii.gz" % (name_, str(radius_))
        ni.save(map_, os.path.join(path,fname))
        
        self._mean_map = return_map


class DecodingSubjectResult(SubjectResult):
    
    _object_attributes = ['classifier', 'mapper', 'ds', 'stats'] 
    
    def save(self, path):
        
        map_mean = self._map.pop()
        map_mean_data = map_mean.get_data()
        map_mean_data = (map_mean_data - np.mean(map_mean_data)) / np.std(map_mean_data)
        fname = "%s_mean_map.nii.gz" % self.name
        img_zscore = ni.Nifti1Image(map_mean_data, map_mean.get_affine())
        ni.save(img_zscore, os.path.join(path, fname))
 
        for map_, tar in zip(self._map, self._sensitivities.sa.targets):
            classes_ = '_'.join(tar)
            fname = "%s_%s_map.nii.gz" % (self.name, classes_)
            map_data = map_.get_data()
            map_data_zscore = (map_data - np.mean(map_data)) / np.std(map_data)
            map_zscore = ni.Nifti1Image(map_data_zscore, map_.get_affine())
            ni.save(map_zscore, os.path.join(path,fname))
                
        # Save statistics
        stats = self._stats
        fname = "%s_stats.txt" % self.name
        file_ = open(os.path.join(path,fname), 'w')
        p_value = self._pvalue
        file_.write(str(stats))
        file_.write('\n\np-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        
        file_.write('\n\nMean each fold p-value: '+str(p_value.mean()))
        
        file_.close()
        
        for obj_name in self._object_attributes:
            obj = getattr(self, '_'+obj_name)
            if obj_name == 'classifier':
                obj = obj.ca
            fname = "%s_%s.pyobj" % (self.name, obj_name)          
            file_ = open(os.path.join(path,fname), 'w')
            pickle.dump(obj, file_)
            file_.close()


class SimilaritySubjectResult(SubjectResult):
    
    _values = ['distances', 'pvalues']
    
    def __init__(self, name, dict_):
        super(SimilaritySubjectResult, self).__init__(name, dict_)
        
        self._unique_targets = np.unique(self._similarity_data['labels'])
        self._unique_predictions = np.unique(self._similarity_data['predictions'])
        
        self._data = {}
        for p in self._unique_predictions:
            self._data[p] = {}
            for t in self._unique_targets:
                self._data[p][t] = {}
        
  
    def save(self, path):
        
        import matplotlib.pyplot as pl
        # TODO: Keep in mind! They should be saved when similarity has been performed!!!
        similarity_data = self._similarity_data
        similarity_mask = self._similarity_mask
        threshold = self._threshold_value
        
        # Renaming variables to better read
        ds_targets = similarity_data['labels']
        class_prediction_tar = similarity_data['predictions']
        
        fname = '%s_mahalanobis_data.txt' % self.name
        file_ = open(os.path.join(path, fname), 'w')
        
        
        plot_list = dict()
              
                
        for lab in self._unique_predictions:
            
            f, ax = pl.subplots(2, 1)
            plot_list[lab] = [f, ax]
            
            for tar in self._unique_targets: 
            
                # Select data belonging to target loop class
                target_mask = ds_targets == tar
                similarity_target = similarity_mask[target_mask]
                target_data = similarity_data[target_mask] 

                # Select target data classified as loop label
                prediction_mask = target_data['predictions'] == lab 
                crossd_data = target_data[prediction_mask] 
                similarity_crossd = similarity_target[prediction_mask]

                # Filter data that meets similarity criterion
                masked_similarity_data = crossd_data[similarity_crossd]
                num = len(masked_similarity_data)
                
                occurrence_ = [lab, tar, str(num)]
                                
                for i, key_ in enumerate(self._values):
                    
                    key_data = crossd_data[key_]
                    
                    if len(target_data) != 0:
                        key_similarity_data = masked_similarity_data[key_]
                    else:
                        key_similarity_data = np.mean(key_data)
                    
                    key_mean_data = np.mean(key_similarity_data)
                    key_total_data = np.mean(key_data)
                    
                    occurrence_.append(str(key_mean_data))
                    occurrence_.append(str(key_total_data))
                    
                    # TODO: Maybe is possible to collapse both file in a single one!
                    histo_d_fname = "%s_hist_%s_%s_%s.txt" % (self.name, lab, tar, key_)
                    np.savetxt(os.path.join(path, histo_d_fname), np.float_(key_data))
                    
                    # Histogram plots
                    # TODO: Maybe it's better to do something else!
                    # TODO: Unique values of bins!
                    plot_list[lab][1][i].hist(np.float_(key_data), bins=35, label=tar, alpha=0.5)
                    self._data[lab][tar][key_] = np.float_(key_data)
                    
                    if key_ == 'pvalues':
                        continue
                    
                    
                    ## TODO: Insert plot in a function and let the user decide if he wants it!
                    total_data = getattr(self, '_'+key_)
                    data = total_data[lab][target_mask]
                    
                    f_d = pl.figure()
                    
                    a_d = f_d.add_subplot(111)
                    a_d.plot(data)
                    a_d.set_ylim(data.mean()-3*data.std(), data.mean()+3*data.std())
                    step = data.__len__() / 6.
                    for j in np.arange(6)+1:#n_runs
                        a_d.axvline(x = step * j, ymax=a_d.get_ylim()[1], color='y', linestyle='-', linewidth=1)
                    a_d.axhline(y = threshold, color='r', linestyle='--', linewidth=2)
                    
                    a_d.axhline(y = np.mean(data), color='black', linestyle=':', linewidth=2)
                    
                    pname = "%s_%s_plot_%s_%s.png" % (self.name, key_, lab, tar)
                    f_d.savefig(os.path.join(path, pname))
                    
                    fname = "%s_%s_txt_%s_%s.txt" % (self.name, key_, lab, tar)
                    np.savetxt(os.path.join(path, fname), 
                               data, fmt='%.4f')               

                    #means_s[lab+'_'+tar].append(np.mean(data))
                    
                write_string = ','.join(occurrence_)
                file_.write(write_string+'\n')              

 
            ## TODO: Insert in a function                
            ax1 = plot_list[lab][1][0]
            ax2 = plot_list[lab][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            fig_name = os.path.join(path, self.name+'_histogram_'+lab+'.png')
            plot_list[lab][0].savefig(fig_name)

        file_.write('\nthreshold '+str(threshold))       
        file_.close()
        
        cmatrix_mahala = self._confusion_mahalanobis
        fname = "%s_confusion_mahala.txt" % self.name
        file_ = open(os.path.join(path ,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            print err
        
        file_.close()



class SignalDetectionSubjectResult(SubjectResult):
    
    _indexes = ['beta', 'c', 'd_prime']
    
    def __init__(self, name, dict_):
        super(SignalDetectionSubjectResult, self).__init__(name, dict_)
        
    def save(self, path):
        
        fname = '%s_signal_detection_indexes.txt' % (self.name)
        file_ = open(os.path.join(path, fname), 'w')
        
        for key_ in self._indexes:
            file_.write(key+','+str(getattr(self, '_'+key_)))
            file_.write("\n")
        
        file_.close()





def make_dir(path):
    """ Make dir unix command wrapper """
    command = 'mkdir '+os.path.join(path)
    os.system(command)
        
def save_image(filename, image):
    return  

def get_time():
    """Get the current time and returns a string (fmt: yymmdd_hhmmss)"""
    
    # Time acquisition
    tempo = time.localtime()
    
    datetime = ''
    i = 0
    for elem in tempo[:-3]:
        i = i + 1
        if len(str(elem)) < 2:
            elem = '0'+str(elem)
        if i == 4:
            datetime += '_'
        datetime += str(elem)
        
    return datetime 


