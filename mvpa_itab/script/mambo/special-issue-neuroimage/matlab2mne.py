################################
# This is to download data from Amazon WS
# $ s3cmd get --recursive s3://hcp.aws.amazon.com/hcp/106521/T1w/106521/
################################

import hcp
import os
import numpy as np

hcp_path = "/media/robbis/DATA/meg/hcp/"

subject = '106521'
hcp_task = 'task_motor'
recordings_path = os.path.join(hcp_path, 'recordings')
fs_path = os.path.join(hcp_path, 'subjects')
hcp.make_mne_anatomy(subject, 
                     subjects_dir=fs_path, 
                     hcp_path=hcp_path, 
                     recordings_path=recordings_path)


epochs = hcp.read_epochs(subject, hcp_task, onset='resp', hcp_path=hcp_path)
info = hcp.read_info(subject=subject, 
                     hcp_path=hcp_path, 
                     data_type=hcp_task,
                     run_index=0)

# 
#    cfg=[];
#    cfg.method  = 'mtmfft';
#    cfg.channel = 'MEG';
#    cfg.trials  = 1;
#    cfg.output  = 'powandcsd'; % gives power and cross-spectral density matrices
#    cfg.foi = 22;
#    cfg.tapsmofrq = 8; %this is 'in both directions' ie nhz up and nhz down
#    cfg.taper = 'dpss';
#    cfg.keeptapers = 'no';
#    freq = ft_freqanalysis(cfg,data);
#

from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper
csd = psd_multitaper(epochs, fmax=30, fmin=14)


#    % Prepare leadfield        
#    cfg=[];
#    cfg.channel       = ft_channelselection('MEG', freq.label);
#    cfg.grid.pos      = sourcemodel2d.pos;
#    cfg.grid.unit     = sourcemodel2d.unit;
#    cfg.grid.inside   = true(size(sourcemodel2d.pos,1),1);
#    cfg.grid.coordsys = sourcemodel2d.coordsys;
#    cfg.vol           = headmodel;
#    cfg.grad          = freq.grad;
#    cfg.reducerank    = 2;
#    leadfield2d       = ft_prepare_leadfield(cfg);

import mne
head_mri_t = mne.read_trans(
    os.path.join(recordings_path, subject, '{}-head_mri-trans.fif'.format(
            subject)))

# Source space
src = mne.setup_source_space(
    subject=subject, subjects_dir=fs_path, add_dist=False,
    spacing='oct6')

# This is to morph the fsaverage source model into subjects.
src_subject = mne.morph_source_spaces(src_fsaverage, 
                                      subject, 
                                      subjects_dir=fs_path)
# BEM
bems = mne.make_bem_model(subject, 
                          conductivity=(0.3,),
                          subjects_dir=fs_path,
                          ico=4)
bem_sol = mne.make_bem_solution(bems)


picks = mne.pick_types(info, meg=True, ref_meg=False)
info = mne.pick_info(info, picks)

# Forward
fwd = mne.make_forward_solution(info, 
                                trans=head_mri_t, 
                                bem=bem_sol,
                                src=src)



