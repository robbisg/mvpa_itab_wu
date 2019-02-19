import nibabel as ni
import os

level_map = {"1":"1", "3":"2", "5":"3"}


def carlo_memory():
    
    path = "/media/robbis/DATA/fmri/memory/0_results/searchlight_analysis/leave_one_subject_out/"
    filelist = os.listdir("/media/robbis/DATA/fmri/memory/0_results/searchlight_analysis/leave_one_subject_out/")
    filelist = [f for f in filelist if f.find("minus") != -1]
    
    command = "Subj  group  level  InputFile "
    
    for f in filelist:
        img = ni.load(path+f)
        n_subjects = img.shape[-1]
        level = f.split("_")[4]
        for i in range(n_subjects):
            if i < 12:
                group = "group_1"
            else:
                group = "group_2"
            
            command += "s%02d  %s  level_%s  %s[%s] " % (i+1, group, level, f, str(i))
            
    return command



def carlo_ofp():
    
    path = "/media/robbis/DATA/fmri/carlo_ofp/0_results/control_analysis/"
    filelist = os.listdir(path)
    filelist = [f for f in filelist if f.find("minus_chance") != -1 and f.find("nii") != -1]
    filelist.sort()
    
    levels = {'1': 0.7, '2': 1.23, '3': 1.68}
    
    command = "Subj  level  InputFile "
    
    for f in filelist:
        img = ni.load(os.path.join(path, f))
        n_subjects = img.shape[-1]
        level = f.split("_")[-1][:-7]
        for i in range(n_subjects):
            
            command += "s%02d  %s  %s[%s] " % (i+1, str(levels[level]), f, str(i))
            
    return command

        

def carlo_memory_within(path, test="omnibus", task="memory", type=""):
    
    filelist = os.listdir(path)
    filelist = [f for f in filelist if f.find(task) != -1 and f.find(type) != -1]
    filelist = [f for f in filelist if f.find(".nii.gz") != -1]
    filelist.sort()
    
    command = "Subj  group  level  InputFile "

    level_string = ""
    if test == "omnibus":
        level_string = "level_"
    
    for f in filelist:
        img = ni.load(os.path.join(path,f))
        n_subjects = img.shape[-1]
        level = f.split("_")[4]
        group = f.split("_")[-1][0]
        for i in range(n_subjects):
            subj_no = (int(group) - 1) * n_subjects + i
            command += "s%02d  group_%s  %s%s  %s[%s] " % (subj_no+1, group, level_string, level, f, str(i))
            
    return command


3dLME -model 'group*level' \
-prefix level_effect_memory_minus_chance \
-qVars level \
-ranEff ~1 \
-SS_type 3 \
-num_glt 2 \
-gltLabel 1 t-test-1 -gltCode 1 group : '1*group1' '-1*group2' \
-gltLabel 2 t-test-2 -gltCode 2 level : \
-mask '/media/robbis/DATA/fmri/memory/mask_intersection.nii.gz' \
-dataTable Subj  group  level  InputFile s01  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[0] s02  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[1] s03  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[2] s04  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[3] s05  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[4] s06  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[5] s07  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[6] s08  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[7] s09  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[8] s10  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[9] s11  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[10] s12  group1  1  group1_memory_evidence_1_total_minus_chance.nii.gz[11] s01  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[0] s02  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[1] s03  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[2] s04  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[3] s05  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[4] s06  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[5] s07  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[6] s08  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[7] s09  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[8] s10  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[9] s11  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[10] s12  group1  3  group1_memory_evidence_3_total_minus_chance.nii.gz[11] s01  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[0] s02  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[1] s03  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[2] s04  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[3] s05  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[4] s06  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[5] s07  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[6] s08  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[7] s09  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[8] s10  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[9] s11  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[10] s12  group1  5  group1_memory_evidence_5_total_minus_chance.nii.gz[11] s13  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[0] s14  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[1] s15  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[2] s16  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[3] s17  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[4] s18  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[5] s19  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[6] s20  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[7] s21  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[8] s22  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[9] s23  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[10] s24  group2  1  group2_memory_evidence_1_total_minus_chance.nii.gz[11] s13  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[0] s14  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[1] s15  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[2] s16  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[3] s17  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[4] s18  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[5] s19  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[6] s20  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[7] s21  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[8] s22  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[9] s23  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[10] s24  group2  3  group2_memory_evidence_3_total_minus_chance.nii.gz[11] s13  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[0] s14  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[1] s15  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[2] s16  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[3] s17  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[4] s18  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[5] s19  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[6] s20  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[7] s21  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[8] s22  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[9] s23  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[10] s24  group2  5  group2_memory_evidence_5_total_minus_chance.nii.gz[11]


3dLME \
-model 'group*level' \
-ranEff ~1 \
-prefix omnibus_evidence_memory_minus_chance \
-SS_type 3 \
-num_glf 2 \
-mask '/media/robbis/DATA/fmri/memory/mask_intersection.nii.gz' \
-glfLabel 1 f-test-1 -glfCode 1 level : '1*level_1' '&' '1*level_3' '&' '1*level_5' \
-glfLabel 2 f-test-2 -glfCode 2 level : '1*level_1'  '-1*level_3' '&' '1*level_3' '-1*level_5' \
-dataTable Subj  group  level  InputFile s01  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[0] s02  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[1] s03  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[2] s04  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[3] s05  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[4] s06  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[5] s07  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[6] s08  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[7] s09  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[8] s10  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[9] s11  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[10] s12  group1  level_1  group1_memory_evidence_1_total_minus_chance.nii.gz[11] s01  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[0] s02  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[1] s03  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[2] s04  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[3] s05  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[4] s06  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[5] s07  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[6] s08  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[7] s09  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[8] s10  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[9] s11  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[10] s12  group1  level_3  group1_memory_evidence_3_total_minus_chance.nii.gz[11] s01  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[0] s02  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[1] s03  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[2] s04  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[3] s05  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[4] s06  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[5] s07  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[6] s08  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[7] s09  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[8] s10  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[9] s11  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[10] s12  group1  level_5  group1_memory_evidence_5_total_minus_chance.nii.gz[11] s13  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[0] s14  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[1] s15  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[2] s16  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[3] s17  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[4] s18  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[5] s19  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[6] s20  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[7] s21  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[8] s22  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[9] s23  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[10] s24  group2  level_1  group2_memory_evidence_1_total_minus_chance.nii.gz[11] s13  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[0] s14  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[1] s15  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[2] s16  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[3] s17  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[4] s18  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[5] s19  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[6] s20  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[7] s21  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[8] s22  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[9] s23  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[10] s24  group2  level_3  group2_memory_evidence_3_total_minus_chance.nii.gz[11] s13  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[0] s14  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[1] s15  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[2] s16  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[3] s17  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[4] s18  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[5] s19  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[6] s20  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[7] s21  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[8] s22  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[9] s23  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[10] s24  group2  level_5  group2_memory_evidence_5_total_minus_chance.nii.gz[11]

  
 3dLME \
-model 'group*level' \
-ranEff ~1 \
-prefix omnibus_mdm_group \
-SS_type 3 \
-num_glf 1 \
-mask '/media/robbis/DATA/fmri/memory/mask_intersection.nii.gz' \
-gltLabel 1 t-test-1 -gltCode 1 group : '1*group_1 -1*group_2' \
-dataTable Subj  group  level  InputFile s13  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[0] s14  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[1] s15  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[2] s16  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[3] s17  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[4] s18  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[5] s19  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[6] s20  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[7] s21  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[8] s22  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[9] s23  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[10] s24  group_2  level_1  sl_group_decision_evidence_1_split_1_train_1_test_2.nii.gz[11] s01  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[0] s02  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[1] s03  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[2] s04  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[3] s05  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[4] s06  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[5] s07  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[6] s08  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[7] s09  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[8] s10  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[9] s11  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[10] s12  group_1  level_1  sl_group_decision_evidence_1_split_2_train_2_test_1.nii.gz[11] s13  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[0] s14  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[1] s15  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[2] s16  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[3] s17  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[4] s18  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[5] s19  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[6] s20  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[7] s21  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[8] s22  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[9] s23  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[10] s24  group_2  level_3  sl_group_decision_evidence_3_split_1_train_1_test_2.nii.gz[11] s01  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[0] s02  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[1] s03  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[2] s04  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[3] s05  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[4] s06  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[5] s07  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[6] s08  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[7] s09  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[8] s10  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[9] s11  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[10] s12  group_1  level_3  sl_group_decision_evidence_3_split_2_train_2_test_1.nii.gz[11] s13  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[0] s14  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[1] s15  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[2] s16  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[3] s17  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[4] s18  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[5] s19  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[6] s20  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[7] s21  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[8] s22  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[9] s23  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[10] s24  group_2  level_5  sl_group_decision_evidence_5_split_1_train_1_test_2.nii.gz[11] s01  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[0] s02  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[1] s03  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[2] s04  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[3] s05  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[4] s06  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[5] s07  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[6] s08  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[7] s09  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[8] s10  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[9] s11  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[10] s12  group_1  level_5  sl_group_decision_evidence_5_split_2_train_2_test_1.nii.gz[11]


3dLME \
-model 'group*level' \
-ranEff ~1 \
-prefix ttest \
-SS_type 3 \
-num_glf 1 \
-mask /media/robbis/DATA/fmri/memory/mask_intersection.nii.gz \
-glfLabel 1 f-test-1 -glfCode 1 'group : 1*group_1 & 1*group_2 level : 1*level_1 & 1*level_3 & 1*level_5' \
-dataTable Subj group level InputFile s01 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[0]' s02 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[1]' s03 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[2]' s04 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[3]' s05 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[4]' s06 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[5]' s07 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[6]' s08 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[7]' s09 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[8]' s10 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[9]' s11 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[10]' s12 group_1 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[11]' s13 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[12]' s14 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[13]' s15 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[14]' s16 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[15]' s17 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[16]' s18 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[17]' s19 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[18]' s20 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[19]' s21 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[20]' s22 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[21]' s23 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[22]' s24 group_2 level_1 'sl_group_decision_evidence_1_minus_0-5.nii.gz[23]' s01 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[0]' s02 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[1]' s03 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[2]' s04 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[3]' s05 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[4]' s06 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[5]' s07 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[6]' s08 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[7]' s09 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[8]' s10 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[9]' s11 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[10]' s12 group_1 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[11]' s13 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[12]' s14 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[13]' s15 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[14]' s16 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[15]' s17 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[16]' s18 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[17]' s19 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[18]' s20 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[19]' s21 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[20]' s22 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[21]' s23 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[22]' s24 group_2 level_3 'sl_group_decision_evidence_3_minus_0-5.nii.gz[23]' s01 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[0]' s02 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[1]' s03 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[2]' s04 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[3]' s05 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[4]' s06 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[5]' s07 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[6]' s08 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[7]' s09 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[8]' s10 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[9]' s11 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[10]' s12 group_1 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[11]' s13 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[12]' s14 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[13]' s15 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[14]' s16 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[15]' s17 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[16]' s18 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[17]' s19 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[18]' s20 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[19]' s21 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[20]' s22 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[21]' s23 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[22]' s24 group_2 level_5 'sl_group_decision_evidence_5_minus_0-5.nii.gz[23]'

path = "/media/robbis/DATA/fmri/carlo_mdm/%s/RESIDUALS_MVPA/"
for subj in subjects:
    path_subj = path %(subj)
    outpath = path_subj+"attributes_residuals.txt"
    list_file = os.listdir(path_subj)
    txt_attr = [f for f in list_file if f.find("4CLA_ASS_single_trials") != -1]
    if subj == "110929angque":
        fidl2txt_2(path_subj+txt_attr[0], outpath, vol_run=250, stim_tr=9, offset_tr=1)
    else:
        fidl2txt_2(path_subj+txt_attr[0], outpath, stim_tr=9, offset_tr=1)
    enhance_attributes_memory(outpath)


        
for subj in subjects:
    path_subj = path %(subj)
    outpath = path_subj+"attributes_residuals.txt"
    attributes = np.loadtxt(outpath, dtype=np.str)
    print(subj)
    print(attributes.shape)
    fname = os.path.join(path_subj, "%s_task_Ass_15_10_res_b%02d.nii" %(subj[6:], i+1))
    sum_ = [ni.load(fname).shape[-1] for i in range(12)]
    print(sum_)
    print(np.sum(sum_))
