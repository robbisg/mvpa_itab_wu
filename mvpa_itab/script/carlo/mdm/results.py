import nibabel as ni
import os
import numpy as np
import seaborn as sns
import pandas as pd


path = "/media/robbis/DATA/fmri/memory/"

#### Omnibus ROI evidence
import configparser
config = configparser.ConfigParser()
config.read("/media/robbis/DATA/fmri/memory/0_results/searchlight_analysis/leave_one_subject_out/stats/roi_labels.cod")
roi_names = dict(config.items('across_omnibus_p-0.0008_q-0.01_vx_20_mask'))


roi_mask = ni.load(os.path.join(path, 
            "0_results/searchlight_analysis/across_subjects/", 
            "across_omnibus_p-0.0008_q-0.01_vx_20_mask.nii.gz"))

evidences_fname = os.path.join(path, 
            "0_results/searchlight_analysis/leave_one_subject_out/", 
            "sl_group_decision_evidence_%s_minus_0-5.nii.gz")

evidence_maps = {m:ni.load(evidences_fname % (m)) for m in ['1', '3', '5']}

mask_data = roi_mask.get_data()

data_evidence = {}
full_data = []
for roi_value in np.unique(mask_data)[1:]:
    for evidence in list(evidence_maps.keys()):

        map_data = evidence_maps[evidence].get_data()

        average = map_data[mask_data == roi_value].mean(0)

        data_evidence['roi'] = roi_names[str(roi_value)]
        data_evidence['level'] = evidence

        for subj, avg in enumerate(average):

            data_evidence['subject'] = 'subj%02d' % (subj+1)
            data_evidence['accuracy'] = avg + .535

            full_data.append(data_evidence.copy())

dataframe = pd.DataFrame(full_data)

g = sns.catplot(x="level", y="accuracy", col="roi", col_wrap=4, data=dataframe, kind='point')


from pyitab.analysis.results import get_results

dataframe = get_results('/media/robbis/DATA/fmri/carlo_mdm/0_results', 
                         dir_id="omnibus", 
                         field_list=['cv', 'sample_slicer'],
                         )

g = sns.catplot(x="accuracy", 
                y="score_accuracy", 
                col="roi_value", 
                col_wrap=4, 
                data=dataframe, 
                kind='point')


###### Temporal decoding #######
dataframe = get_results('/media/robbis/DATA/fmri/carlo_mdm/0_results', 
                         dir_id="temporal_omnibus", 
                         field_list=['sample_slicer'],
                         )

# Mean across-folds
df_ = df_fx_over_keys(dataframe=dataframe, 
                      keys=['evidence', 'roi_value', 'subject'], 
                      attr='score_score', 
                      fx=lambda x: np.mean(np.dstack(x), axis=2))

# Mean across-subjects
df_evidence = df_fx_over_keys(dataframe=df_, 
                              keys=['evidence', 'roi_value'], 
                              attr='score_score', 
                              fx=lambda x: np.mean(np.dstack(x), axis=2))

df_diagonal = df_fx_over_keys(dataframe=df_, 
                              keys=['evidence', 'roi_value'], 
                              attr='score_score', 
                              fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))


unique_evidences = np.unique(df_evidence.evidence.values)
unique_roi_val   = np.unique(df_evidence.roi_value.values)


labels = [
            "Caudate",
            "L IPL",
            "L Pre Cent G",
            "R IPL",
            "Med Front G",
            "L Claustrum",
            "L Sup Temp G",
            "L Cuneus",
            "R Mid Front G",
            "L Med Front G",
            "L Sup Temp G II",
         ]


dataframe_list = []
for ev in unique_evidences:
    for r in unique_roi_val:
        filtered_df = filter_dataframe(df_diagonal, evidence=[ev], roi_value=[r])
        new_field = {"evidence":ev, "roi_value":r, "label":labels[int(r)-1]}

        for i, accuracy in enumerate(filtered_df['score_score'].values[0].tolist()):
            new_field.update({"score_score":accuracy, "frame":i+1})

            dataframe_list.append(new_field.copy())

df_column_diag = pd.DataFrame(dataframe_list)





############



unique_evidences = ['1', '3', '5']
unique_roi_val   = np.unique(df_evidence.roi_value.values)


fig1, ax1 = pl.subplots(len(unique_evidences), len(unique_roi_val), sharex=True, sharey=True)
fig2, ax2 = pl.subplots(len(unique_evidences), len(unique_roi_val), sharex=True, sharey=True)

[ax1[0,j].set_title(labels[j]) for j in range(11)]
[ax2[0,j].set_title(labels[j]) for j in range(11)]
[ax1[3,j].set_xlabel('frame') for j in range(11)]
[ax2[3,j].set_xlabel('frame') for j in range(11)]

[ax1[i,0].set_ylabel(unique_evidences[i]) for i in range(4)]
[ax2[i,0].set_ylabel(unique_evidences[i]) for i in range(4)]

for i, ev in enumerate(unique_evidences):
    for j, r in enumerate(unique_roi_val):

        filtered_df = filter_dataframe(df_evidence, evidence=[ev], roi_value=[r])
        matrix = filtered_df['score_score'].values[0]

        p = ax2[i,j].plot(np.diagonal(matrix), 'o-')
        im = ax1[i,j].imshow(matrix, origin='lower', vmin=0.5, vmax=0.65, cmap='magma')

        ax2[i,j].set_ylim(0.45, 0.65)


for ax in ax2.flat:
    ax.set(xlabel='frame', ylabel='accuracy')

for ax in ax1.flat:
    ax.set(xlabel='testing-frame', ylabel='training-frame')

for ax in ax2.flat:
    ax.label_outer()

for ax in ax1.flat:
    ax.label_outer()