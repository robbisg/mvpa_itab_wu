from mvpa2.generators.partition import CustomPartitioner


class SKLCrossValidation(CustomPartitioner):
    
    def __init__(self, sklearn_cv_obj, **kwargs):
        
        splitrule = []
        for _, test in sklearn_cv_obj:
            splitrule+=[(None, test)]
            
        CustomPartitioner.__init__(self, splitrule, **kwargs)
