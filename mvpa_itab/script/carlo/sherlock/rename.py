import os
from pyitab.utils.image import afni_converter


path = "/home/robbis/mount/permut1/sherlock/"
subjects = os.listdir(path)
subjects = [s  for s in subjects if s[0].isnumeric()]
subjects.sort()
fields = (s[:6], s[6:-5], s[-4:])


for s in subjects:
    day, subject, task = s[:6], s[6:-5], s[-4:]

    folder = os.path.join(path, "sub-%s" %(subject), "func")
    print("mkdir -p %s" % (folder))
    os.system("mkdir -p %s" % (folder))

    # Files
    files = os.listdir(os.path.join(path, s, "prepro"))
    files.sort()

    fx = np.equal
    if task == 'day1':
        fx = np.not_equal

    #files = [f for f in files if f.find("tlrc_al+tlrc.HEAD") != -1 and fx(f.find("cut"), -1)]
    files = [f for f in files if f.find("tlrc_al_scale.nii.gz") != -1]

    for f in files:
        input_fname = os.path.join(path, s, "prepro", f)
        run = int(f.split("_")[2][1:])

        output_name = "sub-%s_task-%s_run-%02d_space-MNI_bold.nii.gz" % (
            subject, task, run
        )

        output_fname = os.path.join(folder, output_name)
        #print(output_fname)
        mask_fname = os.path.join(path, s, "prepro", "%s_mask_epi_MNI.nii.gz" %(s))
        #command = "3dcalc -a %s -b %s -expr 'a*b' -prefix %s" % (input_fname[:-5], mask_fname, output_fname)
        command = "3dcalc -a %s -b %s -expr 'a*b' -prefix %s" % (input_fname, mask_fname, output_fname)
        print(command)
        os.system(command)

####################################
# Events
path_events = "/home/robbis/Downloads/"
excel_fname = os.path.join(path_events, "s%02d_%s_ROB.xlsx")

encoding_names = ['run', 'onset', 'trial_type', 'duration']
encoding_order = ['onset', 'duration', 'trial_type', 'run']

retrieva_names = ['run', 'onset', 'trial_type', 'dimension', 'response',
                  'accuracy', 'confidence', 'duration']
retrieva_order = ['onset', 'duration', 'trial_type', 'dimension', 'response',
                  'accuracy', 'confidence', 'run']


path = "/home/robbis/mount/permut1/sherlock/"
subjects = os.listdir(path)
subjects = [s for s in subjects if s[0].isnumeric()]




for i, s in enumerate(subjects):
    day, subject, task = s[:6], s[6:-5], s[-4:]
    output_folder = os.path.join(path, "bids","derivatives", "sub-%s" %(subject), "func")

    if task == 'day1':
        task_exc = 'ENC'
        names = encoding_names
        order = encoding_order
    else:
        task_exc = 'RET'
        names = retrieva_names
        order = retrieva_order

    events = pd.read_excel(excel_fname % (int(i/2)+1, task_exc))
    runs = np.unique(events['RUN'])

    for r in runs:
        mask = events['RUN'] == r
        run_events = events.loc[mask]
        run_events.columns = names
        run_events = run_events.reindex(columns=order)

        event_fname = "sub-%s_task-%s_run-%02d_space-MNI_events.tsv" % (subject, task, int(r))
        print(os.path.join(output_folder, event_fname))
        run_events.to_csv(os.path.join(output_folder, event_fname), sep='\t')

####################################à
# Mask
# 200218matsim_day1_mask_epi_MNI.nii.gz

path = "/home/robbis/mount/permut1/sherlock/"
subjects = os.listdir(path)
subjects = [s  for s in subjects if s[0].isnumeric()]
subjects.sort()

for s in subjects:
    day, subject, task = s[:6], s[6:-5], s[-4:]

    # Files
    files = os.listdir(os.path.join(path, s, "prepro"))
    files.sort()

    files = [f for f in files if f.find("mask_epi_MNI.nii.gz") != -1]

    folder = os.path.join(path, "bids", "derivatives", "preproc", "sub-%s" %(subject), "func")

    output_name = "sub-%s_task-%s_space-MNI_mask.nii.gz" % (
            subject, task
    )

    command = "cp %s %s" % (
        os.path.join(path, s, "prepro", files[0]),
        os.path.join(folder, output_name)
    )
    print(command)
