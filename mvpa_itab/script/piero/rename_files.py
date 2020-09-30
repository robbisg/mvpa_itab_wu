import os

path = "/media/robbis/DATA/fmri/EGG/"
files = os.listdir("/media/robbis/DATA/fmri/EGG/")
tsvpath = os.path.join(path, 'tsv')
tsvfiles = os.listdir(tsvpath)
filtpath = os.path.join(path, 'filtered')
filtfiles = os.listdir(filtpath)
betapath = os.path.join(path, 'betas')
betafiles = os.listdir(betapath)


files = [f for f in files if f.find('.') != -1]

subjects = []
for f in files:
    d = get_dictionary(f)
    subdir = d['sub']
    subjects.append(d['sub'])
    newdir = os.path.join(path, subdir)
    print("mkdir -p %s" %(newdir))
    if 'variant' is in d.keys():
        task = d['variant']
    else:
        task = 'plain'

    taskdir = os.path.join(newdir, task)
    print("mkdir -p %s" %(taskdir))
    
    print("mv %s %s" % (os.path.join(path, d['filename']), taskdir))
    
subjects = np.unique(subjects)

for subdir in subjects:
    newdir = os.path.join(path, subdir)
    filt_filt = [ff for ff in filtfiles if t.find(subdir) != -1]

    print("mkdir -p %s" %(os.path.join(newdir, "filtered")))
    nfilt = os.path.join(newdir, "filtered")
    for ff in filt_filt:
        print("mv %s %s" % (os.path.join(filtpath, ff), 
                            os.path.join(nfilt, ff)
                            ))

    beta_filt = [b for b in betafiles if t.find(subdir) != -1]

    print("mkdir -p %s" %(os.path.join(newdir, "beta")))
    nbeta = os.path.join(newdir, "beta")
    for b in beta_filt:
        print("mv %s %s" % (os.path.join(betapath, b), 
                            os.path.join(nbeta, b)
                            ))

for subdir in subjects:
    filt_tsv = [t for t in tsvfiles if t.find(subdir) != -1]
    for d in ['plain', 'smoothAROMAnonaggr', 'beta', 'filtered']:
        for t in filt_tsv:
            print("cp %s %s/" % (os.path.join(tsvpath, t), 
                                os.path.join(path, subdir, d)
                                ))

#####################################################################################
##################### Refactoring BIDS directory ###################################

def write_description(path, pipeline):
    description =  {
                    "Name": "EGG - Piero %s" %(pipeline),
                    "BIDSVersion": "1.1.1",
                    "PipelineDescription": {
                        "Name": "%s" % (pipeline),
                    },
                    "CodeURL": "https://github.com/robbisg/pyitab"
                }

    dataset_desc = os.path.join(path, 
                                "dataset_description.json")
    
    if not os.path.exists(dataset_desc):
        import json
        
        with open(dataset_desc, 'w') as fp:
            json.dump(description, fp)

    return


def change_dictionary(pipeline, filename):
    dictionary = get_dictionary(filename)
    condition = dictionary['task']

    dictionary['condition'] = condition
    dictionary['task'] = pipeline
    if pipeline == 'beta' or pipeline == 'filtered':
        dictionary['bandpass'] = '004+012'
        if dictionary['filetype'] != 'events':
            dictionary['filetype'] = 'preproc'

    if pipeline == 'smoothAROMAnonaggr':
        if 'variant' in dictionary.keys():
            dictionary.pop('variant')
    
    return dictionary


def write_files(dictionary):

    prefix = "sub-"+dictionary.pop('sub')
    filetype = dictionary.pop('filetype')
    extension = "."+dictionary.pop('extension')
    _ = dictionary.pop('filename')

    filename = "_".join(["%s-%s" % (k, v) for k, v in dictionary.items()])
    filename = prefix + "_" + filename + "_" + filetype + extension
    #print(dictionary, filename)
    return filename



path = "/media/robbis/DATA/fmri/EGG/"
derivatives_path = os.path.join(path, 'derivatives')

os.system("mkdir -p %s" % (derivatives_path))
pipelines = ['plain', 'smoothAROMAnonaggr', 'beta', 'filtered']
count_dictionary = []
for p in pipelines:
    #print("mkdir -p %s" % ())
    #write_description(os.path.join(derivatives_path, p), p)

    for s in subjects:
        pipeline_path = os.path.join(path, s, p)
        files = os.listdir(pipeline_path)

        # make new dir
        subject_dir = os.path.join(derivatives_path, p, "sub-"+s)
        command = "mkdir -p %s" % (subject_dir)
        os.system(command)
        # rename files
        for f in files:
            d = change_dictionary(p, f)
            filename = write_files(d)
            command = "mv %s %s" % (os.path.join(pipeline_path, f), 
                                os.path.join(subject_dir, filename)
                                )
            os.system(command)
            count_dictionary.append(os.path.join(subject_dir, filename))


