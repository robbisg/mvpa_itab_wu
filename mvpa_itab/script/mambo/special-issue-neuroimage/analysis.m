restoredefaultpath;

clear
clc
close all

% setup FieldTrip
addpath(fullfile('/home/robbis/Programmi/matlab/toolbox/','fieldtrip-20200914'));
ft_defaults;

% setup megconnectome
addpath(genpath(fullfile('/home/robbis/Programmi/matlab/toolbox/','megconnectome')))

data_dir = '/run/user/1000/gvfs/smb-share:server=192.168.30.54,share=meg_data_analisi/';
%data_dir = '/media/robbis/DATA/meg/';
output_dir = '/run/user/1000/gvfs/smb-share:server=192.168.30.54,share=meg_data_analisi/HCP_Motor_Task_analysis';
hcp_dir = 'HCP_Motor_Task';
%hcp_dir = 'hcp';

anatomy_dir='HCP_MEG2_release_anatomy';
%anatomy_dir='hcp';

subj_list = [];
fn = dir(fullfile(data_dir,hcp_dir));
k = 0;
for i=1:length(fn)
    if isempty(strfind(fn(i).name,'.'))
        k=k+1;
        subj_list{k} = fn(i).name;
    end
end
clear fn k i

sess_list = {'10','11'};

% % Load template head model
temp_sourcemodel2d_L = ft_read_headshape('Conte69.L.midthickness.4k_fs_LR.surf.gii');
temp_sourcemodel2d_R = ft_read_headshape('Conte69.R.midthickness.4k_fs_LR.surf.gii');

% load the template cortex (in MNI coordinates)
fn = fullfile('Conte69.L.midthickness.4k_fs_LR.surf.gii');
template_cortex = ft_read_headshape({fn, strrep(fn, '.L.', '.R.')});



%ft_plot_mesh(template_cortex,  'facecolor', 'skin', 'edgecolor', 'black')

%% Analysis

% for isubj = 1:length(subj_list)
 
isubj = 1;
%isess = 1;
    for isess = 1:2

        clearvars -except isubj subj_list data_dir isess sess_list output_dir temp_*  template_cortex *_dir


        disp(['Subject ',num2str(isubj)]);
        windows = [.3, .5];
       
        % load the data and templates

        path_to_data    = fullfile(data_dir, hcp_dir,...
                            subj_list{isubj},'MEG','Motort','tmegpreproc',...
                            strcat(subj_list{isubj},'_MEG_',sess_list{isess},...
                            '-Motort_tmegpreproc_TEMG.mat') );
                       
        path_to_trialinfo    = fullfile(data_dir, hcp_dir,...
                            subj_list{isubj},'MEG','Motort','tmegpreproc',...
                            strcat(subj_list{isubj},'_MEG_',sess_list{isess},...
                            '-Motort_tmegpreproc_trialinfo.mat') );                
        
        
        path_to_sourcemodel2d = fullfile(data_dir, anatomy_dir,...
                              subj_list{isubj},'MEG','anatomy',strcat(subj_list{isubj},...
                              '_MEG_anatomy_sourcemodel_2d.mat'));

        path_to_headmodel = fullfile(data_dir,anatomy_dir,...
                            subj_list{isubj},'MEG','anatomy',strcat(subj_list{isubj},...
                            '_MEG_anatomy_headmodel.mat'));

        load(path_to_data);
        load(path_to_trialinfo);
        load(path_to_headmodel);
        load(path_to_sourcemodel2d);
       
        % ensure consistent units
        data.grad     = ft_convert_units(data.grad,'mm');
        headmodel     = ft_convert_units(headmodel,'mm');
        sourcemodel2d = ft_convert_units(sourcemodel2d,'mm');
       
        Fsample = data.fsample;
       
        cfg = [];
        %  2. Block Stim Code:    1-Left Hand,  2 - Left Foot, 4 - Right Hand. 5 - Right Foot, 6 - Fixation
        cfg.trials = find(trlInfo.lockTrl{strcmpi(trlInfo.lockNames,'TEMG')}(:,7)==0 ); % remove trials with NaNs    
        data = ft_selectdata(cfg,data);
       
        %% Source analysis
       
        % frequency analysis
        cfg=[];
        cfg.method  = 'mtmfft';
        cfg.channel = 'MEG';
        cfg.trials  = 1;
        cfg.output  = 'powandcsd'; % gives power and cross-spectral density matrices
        cfg.foi = 22;
        cfg.tapsmofrq = 8; %this is 'in both directions' ie nhz up and nhz down
        cfg.taper = 'dpss';
        cfg.keeptapers = 'no';
        freq = ft_freqanalysis(cfg,data);

        % Prepare leadfield        
        cfg=[];
        cfg.channel       = ft_channelselection('MEG', freq.label);
        cfg.grid.pos      = sourcemodel2d.pos;
        cfg.grid.unit     = sourcemodel2d.unit;
        cfg.grid.inside   = true(size(sourcemodel2d.pos,1),1);
        cfg.grid.coordsys = sourcemodel2d.coordsys;
        cfg.vol           = headmodel;
        cfg.grad          = freq.grad;
        cfg.reducerank    = 2;
        leadfield2d       = ft_prepare_leadfield(cfg);

        cfg=[];
        cfg.method      = 'eloreta';
        cfg.grid        = leadfield2d;
        cfg.vol         = headmodel;
        cfg.keepfilter  = 'yes';
        cfg.keepmom     = 'no';
        cfg.eloreta.keepfilter  = 'yes';
        cfg.eloreta.keepmom     = 'no';
        cfg.eloreta.lambda      = 0.05;        
        source = ft_sourceanalysis(cfg, freq);
       
        %%
        Wfilt = cat(1,source.avg.filter{:});
        Wfilt_label = source.cfg.channel;
       
        [ndum,nchan] = size(Wfilt);
        nso = ndum/2;
        ntrials = length(data.trial);
             
        trialvec = data.trialinfo(:,2);
        trialdesccode = trlInfo.trlColDescr{1}{2};
        trailinfo = data.trialinfo;
       
        timevec = -0.15:0.02:1;
        ntime   = length(timevec);
       
        powerbox = zeros(nso,ntime,ntrials);

        %%
        labelnew = cell(size(Wfilt,1),1);
        for i=1:size(Wfilt,1)
            labelnew{i} = ['s',num2str(i)];
        end
        
        for w = 1:length(windows)
        
        
            for itrl = 1:ntrials
                itrl

                cfg = [];
                cfg.trials  = itrl;
                cfg.channel = 'MEG';
                data_loc    = ft_selectdata(cfg,data);

                 % ensure correct channel order
                [sel1, sel2] = match_str(data_loc.label, Wfilt_label);

                montage = [];
                montage.tra      = Wfilt(:,sel2);
                montage.labelorg = data_loc.label;
                montage.labelnew = labelnew;
                sodata_loc = ft_apply_montage(data_loc,montage);

                % Time-Frequency Representation (TFR)

                cfg = [];
                cfg.method     = 'mtmconvol';
                cfg.output     = 'pow';
                cfg.foi        = 22;
                cfg.tapsmofrq  = 8;
                cfg.taper      = 'dpss';
                cfg.t_ftimwin  = windows(w);
                cfg.toi        = timevec;
                TFRloc = ft_freqanalysis(cfg, sodata_loc);

                powerbox(:,:,itrl) = ( squeeze(TFRloc.powspctrm(1:2:end,:,:)) + ...
                                       squeeze(TFRloc.powspctrm(2:2:end,:,:)) )/2;
            end
       
 %%      
       
        timevec = TFRloc.time;
        freqvec = TFRloc.freq;
        freqcfg = TFRloc.cfg;
       
        outpn = fullfile(output_dir,subj_list{isubj});
        outpn = '/media/robbis/DATA/c2b/meeting-december-data/';
        if not( exist(outpn,'dir')==7 )
            mkdir(outpn);
        end
        outfn = ['sub-',subj_list{isubj},'_ses-',sess_list{isess},'_window-',num2str(windows(w)*1000),'_powerbox_beta.mat'];
        save(fullfile(outpn,outfn),'powerbox','trialvec','trialdesccode', ...
             'trailinfo','timevec','freqvec','freqcfg','-v7.3')
       
       
      end
    end