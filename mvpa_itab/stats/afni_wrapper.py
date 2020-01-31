import os
import nibabel as ni

class AFNIStats(object):
    
    def __init__(self, command, posthoc, **kwargs):
        """
        Build a command line for afni stats tests 3dLME, 3dMVM (only).
        Users should init the class and then transform set_posthoc and set_data_table, 
        finally execute transform function
        
        >> lme = AFNIStats(command="3dLME", 
        >>                 posthoc={"f": ["level : 1*level_1 & 1*level_2 & 1*level_3"]}, 
        >>                 prefix="omnibus_evidence_ofp", model="level", ranEff='~1', SS_type="3")
        
        >> lme.set_posthoc()
        >> lme.set_datatable(carlo_ofp)
        >> lme.command
        
        Parameters
        ----------
        command : string
            The afni binary command to be executed (3dLME or 3dMVM)
        
        posthoc : dictionary
            The posthoc to be calculated by the afni program.
            The dictionary must be in the form of test, list of contrast: 
            the test is a t-test or f-test and must be named "t" or "f", the list
            of contrasts must be a list of strings that express the null hypothesis to test
            Read the afni command guide to understand how they should be written.
            
        kwargs : dictionary of command options
            These are the parameters of the function please refer to afni documentation
            to a full list of them
            
        Returns
        -------

        """        
        
        self.command = command
        
        for k, v in kwargs.iteritems():
            self.k = v
            self.command += " -%s '%s'" %(k, v)
            
            
        self.posthoc = posthoc
        
    
    def set_posthoc(self, posthoc=None):
        
        if posthoc != None:
            self.posthoc = posthoc
            
        for test, contrasts in self.posthoc.iteritems():
            self.command += " -num_gl%s %s" % (test, len(contrasts))
            
            for i, c in enumerate(contrasts):
                self.command += " -gl%sLabel %s '%s-test-%s' -gl%sCode %s '%s'" %(test, 
                                                                                  str(i+1), 
                                                                                  test, 
                                                                                  str(i+1), 
                                                                                  test, 
                                                                                  str(i+1), 
                                                                                  c)
                
                
            
    def set_datatable(self, data_table_function):
        """
        This function takes in input a self-consistent function
        that should be used to output the datatable in afni format
        please refer to afni docs
        """
        
        data_table = data_table_function()
        self.command += " -dataTable "+data_table
        
    
    def transform(self):
        os.system(self.command)
        
        
        

class MVM(AFNIStats):
    
    def __init__(self, posthoc, **kwargs):
        AFNIStats.__init__(self, command="3dMVM", posthoc=posthoc, **kwargs)
        
        

class LME(AFNIStats):
    
    def __init__(self, posthoc, **kwargs):
        AFNIStats.__init__(self, command="3dLME", posthoc=posthoc, **kwargs)
        
        
        

class ANOVA(AFNIStats):
    
    def __init__(self, **kwargs):
        AFNIStats.__init__(self, command="3dANOVA", posthoc=None, **kwargs)
        
    
    def set_datatable(self, levels, level_images):
        
        self.command += " -levels %d "%(len(levels))
        self.levels = len(levels)
                
        
        for i, level in enumerate(levels):
            
            img = ni.load(level_images[i])
            
            for s in range(img.shape[-1]):
                self.command += " -dset %s %s[%s] " %(str(i+1), level_images[i].split("/")[-1], str(s))
                
            self.command += " -mean %s %s " %(str(i+1), level)
            
            
            
    def set_posthoc(self):
        self.command += " -contr 1 1 1 omnibus"
        
        
        
        
        
        
        
        
        
        
        