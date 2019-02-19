import pandas as pd
from pyitab.analysis.results import filter_dataframe
import numpy as np
import os


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

event_file = pd.read_excel("/home/robbis/Downloads/group_4MVPA (1).xls")
event_file['image_cat_num'] = le.fit_transform(df_num['image_cat'].values)


subjects = np.unique(event_file['name'].values)

path_wk = "/home/robbis/mount/meg_workstation/Carlo_MDM/%s/RESIDUALS_MVPA/beta_attributes_full.txt"
folders = os.listdir("/home/robbis/mount/meg_workstation/Carlo_MDM/")

name_dict = {}

for subj in subjects:

    subdir = folders[np.nonzero([s.find(subj) != -1 for s in folders])[0][0]]

    name_dict[subj] = subdir

    df = filter_dataframe(event_file, name=[subj])
    
    df.to_csv(path_wk %(subdir), index=False, sep=" ")



name_dict = {}

for subj in subjects:

    subdir = folders[np.nonzero([s.find(subj) != -1 for s in folders])[0][0]]

    name_dict[subj] = subdir

event_file['name'] = [name_dict[name] for name in event_file['name'].values]



df_num = event_file.copy()

for field in df_num.keys():
    values = le.fit_transform(df_num[field].values)
    #values -= values.mean()
    df_num[field] = values