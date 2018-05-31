import os
import nibabel as ni
import numpy as np
import logging
from scipy.stats import ttest_1samp
import pickle


logger = logging.getLogger(__name__)


class SubjectResult(object):
    
    def __init__(self, name, result_dict, savers=None):
        self.name = name
        
        for k, v in result_dict.items():
            logger.debug("Setting %s" % (k))
            setattr(self, '_'+str(k), v)
        
        if savers != None:
            self.savers = savers
    
    def add_result(self, res_name, value):
        setattr(self, res_name, value)
        
    def set_savers(self, savers):
        "List of objects implementing a save function"
        self.savers = savers


   
class ResultsCollection(object):
    
    
    def __init__(self, conf, path, summarizers=None, **kwargs):
        
        self.collection = []
        self.index = 0
        self.count = 0
        self.summary = dict()
        
        # Get information to make the results dir
        datetime = get_time()
        self._mask = conf['mask_area']
        self._task = conf['analysis_task']
        self._analysis = conf['analysis_type']
        self._conf = conf
        
        self.analysis_path = path
        dir_ = '_'.join([datetime, self._analysis, self._mask, self._task])
        if kwargs != None:
            dir_ += "_"
            for key, value in kwargs.iteritems():
                
                if isinstance(value, dict):
                    
                    for kk, vv in value.iteritems():
                        vv = [str(v) for v in vv]
                        dir_ += str(kk)+"_"+"_".join(vv)
                else:
                    dir_ += str(key)+"_"+str(value)

        self.path = os.path.join(self.analysis_path, '0_results', dir_)
    
        make_dir(self.path)
    
        if summarizers != None:
            self.summarizers = summarizers
            
            
    def set_summarizers(self, summarizers):
        self.summarizers = summarizers
    
    
    def add(self, result):

        if isinstance(result, SubjectResult):
            self.count += 1
            self.collection.append(result)
            subj_path = os.path.join(self.path, result.name)
            logging.debug(subj_path)
            make_dir(subj_path)
            for saver in result.savers:
                saver.save(subj_path, result)
        else:
            print result.__class__
            print type(result)
            print SubjectResult
            
                
    
    def summarize(self):
        #super(ResultsCollection, self).summarize()
        for summarizer in self.summarizers:
            for item in self.collection:
                summarizer.aggregate(item)
            summarizer.summarize(self.path)
        
        import json
        json.dump(self._conf, 
                  file(os.path.join(self.path, 'configuration.json'), 'w'), 
                  indent=-1,
                  sort_keys=True)
        



class Summarizer(object):
    
    ## TODO: Could be useful do a summary header?
    
    def __init__(self):
        self.summary = dict()
  
  
    def aggregate(self, result):
        self.summary[result.name] = []
        return
    
    
    def summarize(self, path):
        # TODO: Use a template
        self.path = path
        
        fname = "analysis_summary.txt"
        file_summary = open(os.path.join(self.path, fname), "a")
        
        file_summary.write(','.join(self.header))
        file_summary.write('\n')
        
        for name, value in self.summary.items():
            file_summary.write(name+',')
            for v in value:
                file_summary.write(str(v)+',')    
            file_summary.write('\n')
        file_summary.close()


               
class SearchlightSummarizer(Summarizer):
    
    expected_fields = ['name', 'map', 'radius']
    
    def __init__(self, p=0.05):
        super(SearchlightSummarizer, self).__init__()
        self._total_map = []        
        self._p_value = p


    def aggregate(self, result):
        
        """
        When we aggregate a result, we store new single subject results
        """
        super(SearchlightSummarizer, self).aggregate(result)
        if not isinstance(result, SubjectResult):
            raise ValueError("Bad class added!")
        
        if not hasattr(self, '_affine'):
            self._affine = result._map.get_affine()
        if not hasattr(self, '_radius'):
            self._radius = result._radius

        self._total_map.append(result._mean_map)

    
    def summarize(self, path):
        self.path = path
        """
        Function who saves the union of subject's map to easily see it
        and the mean map across subjects
        """     
        
        threshold_value = self._p_value
        total_map = self._total_map
        affine = self._affine
        radius = self._radius
            
        total_map = np.rollaxis(np.array(total_map), 0, 4)
        total_img = ni.Nifti1Image(total_map, affine=affine)
        
        fname = "accuracy_map_radius_%s_searchlight_all_subj.nii.gz" % radius
        ni.save(total_img, os.path.join(self.path,fname))
        
        mean_img = ni.Nifti1Image(total_map.mean(3), affine=affine)
        
        fname = "accuracy_map_radius_%s_searchlight_mean_subj.nii.gz" % radius
        ni.save(mean_img, os.path.join(self.path,fname))         
        
        logging.info("Testing versus "+str(self._p_value))
        t, p = ttest_1samp(total_map, threshold_value, axis=3)
        
        fname = "t_map_vs_threshold_%s_uncorrected.nii.gz" % threshold_value 
        _img = ni.Nifti1Image(t, affine=affine)
        ni.save(_img, os.path.join(self.path, fname))
        
        fname = "p_map_vs_threshold_%s_uncorrected.nii.gz" % threshold_value 
        _img = ni.Nifti1Image(p, affine=affine)
        ni.save(_img, os.path.join(self.path, fname))
        
        logging.info('Summarizer writed in '+self.path)



class DecodingSummarizer(Summarizer):
    
    def __init__(self):
        super(DecodingSummarizer, self).__init__()
    
    
    def aggregate(self, result):
        super(DecodingSummarizer, self).aggregate(result)
        self.header = ['name','accuracy', 'pvalue']
        self.summary[result.name].append(result._stats.stats['ACC'])
        self.summary[result.name].append(result._p)
    
    '''
    def summarize(self):
        # TODO: Write a mean map of weights??
    '''   



class SimilaritySummarizer(Summarizer):
        
    def __init__(self):
        super(SimilaritySummarizer, self).__init__()
        self._histogram_data = []
        
        
    def aggregate(self, result):
        super(SimilaritySummarizer, self).aggregate(result)
        
        self._histogram_data.append(result._data)
        self._unique_predictions = result._unique_predictions
        self._unique_targets = result._unique_targets
        self._items = SimilaritySaver._values
        
    def summarize(self, path):
        self.path = path
        
        import matplotlib.pyplot as pl
        
        for t in self._unique_targets:
            fig = pl.figure()
            
            for i, k_ in enumerate(self._items):
                print i, len(self._items)
                ax = fig.add_subplot(len(self._items),1,(i+1))
                
                for p in self._unique_predictions:
                    
                    hist_data = []
                    
                    for data_ in self._histogram_data:
                        hist_data.append(data_[p][t][k_])
                    
                    hist_data = np.hstack(hist_data)
                
                    ax.hist(hist_data, bins=35, label=p, alpha=0.5)
        
        
                ax.legend()
            
            pname = os.path.join(self.path, "total_histogram_%s.png" % (t))
            fig.savefig(pname) 
    
    

class SignalDetectionSummarizer(Summarizer):
    
    def __init__(self):
        super(SignalDetectionSummarizer, self).__init__()
        
    def aggregate(self, result):
        super(SignalDetectionSummarizer, self).aggregate(result)
        self.header = ['name', 'beta', 'd_prime', 'c']
        self.summary[result.name].append(result._beta)
        self.summary[result.name].append(result._d_prime)
        self.summary[result.name].append(result._c)


class CrossDecodingSummarizer(Summarizer):
    
    def __init__(self):
        super(CrossDecodingSummarizer, self).__init__()
                
    def aggregate(self, result):
        super(CrossDecodingSummarizer, self).aggregate(result)
        self.header = ['name','cross_accuracy']
        self.summary[result.name].append(result._confusion_total.stats['ACC'])

  
class Saver(object):
    
    def __init__(self, fields=None):
        self._fields = fields
        return
    
    def save(self, path, result):
        return

        

class CrossDecodingSaver(Saver):
    
    # _fields = ['confusion_target', 'confusion_total']
    
    def __init__(self, fields=['confusion_target', 'confusion_total']):
        return Saver.__init__(self, fields=fields)
            
            
                
    def save(self, path, result):
        
        for k in self._fields:
            matrix = getattr(result, '_'+k)
            fname = "%s_%s.txt" % (result.name, k)
            file_ = open(os.path.join(path, fname), 'w')
            file_.write(str(matrix))
            file_.close()


class SearchlightSaver(Saver):
            
    
    def save(self, path, result):
        """
        When we aggregate a result we store new single subject results
        """
        
        name_ = result.name
        radius_ = np.int(result._radius)
        map_ = result._map
        
                
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
        
        result._mean_map = return_map


class DecodingSaver(Saver):
    
    # _object_attributes = ['classifier', 'mapper', 'ds_src', 'stats']
    
    def __init__(self, fields=['classifier', 'mapper', 'ds_src', 'stats']):
        return Saver.__init__(self, fields=fields)
    
    
    
    def save(self, path, result):
        
        map_mean = result._map.pop()

        save_image(os.path.join(path, "%s_mean_map.nii.gz" % result.name), 
                   map_mean
                   )
 
        for map_, tar in zip(result._map, result._sensitivities.sa.targets):
            classes_ = '_'.join(tar)
            save_image(os.path.join(path, "%s_%s_map.nii.gz" % (result.name, 
                                                                classes_)),
                       map_
                       )
                
        # Save statistics
        stats = result._stats
        fname = "%s_stats.txt" % result.name
        file_ = open(os.path.join(path,fname), 'w')
        file_.write(str(stats))
        """
        p_value = result._perm_pvalue
        file_.write('\n\np-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        
        file_.write('\n\nMean each fold p-value: '+str(p_value.mean()))
        """
        file_.close()
        
        for obj_name in self._fields:
            try:
                obj = getattr(result, '_'+obj_name)
            except AttributeError, err:
                logger.error(err)
                continue
            if obj_name == 'classifier':
                obj = obj.ca
            if obj_name == 'ds_src':
                fname = "%s_ds_stats.txt" % result.name
                file_ = open(os.path.join(path, fname), 'w')
                file_.write(str(obj.summary()))
                file_.close()
            fname = "%s_%s.pyobj" % (result.name, obj_name)          
            file_ = open(os.path.join(path,fname), 'w')
            pickle.dump(obj, file_)
            file_.close()


class SimilaritySaver(Saver):
    
    # _values = ['distances', 'pvalues']
    
    def __init__(self, fields=['distances', 'pvalues']):
        return Saver.__init__(self, fields=fields)
      
            
    def save(self, path, result):
        
        self._get_structures(result)
        
        import matplotlib.pyplot as pl
        # TODO: Keep in mind! They should be saved when similarity has been performed!!!
        similarity_data = result._similarity_data
        similarity_mask = result._similarity_mask
        threshold = result._threshold_value
        
        # Renaming variables to better read
        ds_targets = similarity_data['labels']
        #class_prediction_tar = similarity_data['predictions']
        
        fname = '%s_mahalanobis_data.txt' % result.name
        file_ = open(os.path.join(path, fname), 'w')
        
        
        plot_list = dict()
              
                
        for lab in result._unique_predictions:
            
            f, ax = pl.subplots(2, 1)
            plot_list[lab] = [f, ax]
            
            for tar in result._unique_targets: 
            
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
                                
                for i, key_ in enumerate(self._fields):
                    
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
                    histo_d_fname = "%s_hist_%s_%s_%s.txt" % (result.name, lab, tar, key_)
                    np.savetxt(os.path.join(path, histo_d_fname), np.float_(key_data))
                    
                    # Histogram plots
                    # TODO: Maybe it's better to do something else!
                    # TODO: Unique values of bins!
                    plot_list[lab][1][i].hist(np.float_(key_data), bins=35, label=tar, alpha=0.5)
                    result._data[lab][tar][key_] = np.float_(key_data)
                    
                    if key_ == 'pvalues':
                        continue
                    
                    
                    ## TODO: Insert plot in a function and let the user decide if he wants it!
                    total_data = getattr(result, '_'+key_)
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
                    
                    pname = "%s_%s_plot_%s_%s.png" % (result.name, key_, lab, tar)
                    f_d.savefig(os.path.join(path, pname))
                    
                    fname = "%s_%s_txt_%s_%s.txt" % (result.name, key_, lab, tar)
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
            fig_name = os.path.join(path, result.name+'_histogram_'+lab+'.png')
            plot_list[lab][0].savefig(fig_name)

        file_.write('\nthreshold '+str(threshold))       
        file_.close()
        
        cmatrix_mahala = result._confusion_mahalanobis
        fname = "%s_confusion_mahala.txt" % result.name
        file_ = open(os.path.join(path ,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            logging.error(err)
        
        file_.close()
        
    
    def _get_structures(self, result):
        result._unique_targets = np.unique(result._similarity_data['labels'])
        result._unique_predictions = np.unique(result._similarity_data['predictions'])
        
        result._data = {}
        for p in result._unique_predictions:
            result._data[p] = {}
            for t in result._unique_targets:
                result._data[p][t] = {}



class SignalDetectionSaver(Saver):
    
    # _indexes = ['beta', 'c', 'd_prime']
    
    def __init__(self, fields=['beta', 'c', 'd_prime']):
        return Saver.__init__(self, fields=fields)
       
    def save(self, path, result):
        
        fname = '%s_signal_detection_indexes.txt' % (result.name)
        file_ = open(os.path.join(path, fname), 'w')
        
        for key_ in self._fields:
            file_.write(key_+','+str(getattr(result, '_'+key_)))
            file_.write("\n")
        
        file_.close()



def make_dir(path):
    """ Make dir unix command wrapper """
    #os.mkdir(os.path.join(path))
    command = 'mkdir -p '+os.path.join(path)
    logger.info(command)
    os.system(command)
    
    
def save_image(filename, image, zscore=True):
    
    img_data = image.get_data()
    
    if zscore:
        img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    
    img_save = ni.Nifti1Image(img_data, image.affine)
    ni.save(img_save, filename)  
    
    
    return  



def get_time():
    """Get the current timewise and returns a string (fmt: yymmdd_hhmmss)"""
    
    import time
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


