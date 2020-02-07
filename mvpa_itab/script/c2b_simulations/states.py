
import itertools
import numpy as np
import scipy as sp
from scipy import signal


n_nodes = 10
n_brain_states = 6
max_edges = 5
lifetime_range = [2.5, 3.5] # seconds
n_iteration = 1
sample_frequency = 256


model = PhaseDelayedModel(snr=10)

edges = [e for e in itertools.combinations(np.arange(n_nodes), 2)]
n_edges = len(edges)

n_seq_states = np.floor(60*sample_frequency*5/(sample_frequency*np.mean(lifetime_range)))
n_seq_states = np.int(60*5/np.mean(lifetime_range))

states = []
for i in range(max_edges):
    states += [e for e in itertools.combinations(np.arange(n_edges), i+1)]

states = np.array(states)


for i in range(n_iteration):
    selected_states = states[np.random.randint(0, len(states), n_brain_states)]
    length_bs = np.random.randint(sample_frequency*lifetime_range[0], 
                                  sample_frequency*lifetime_range[1], 
                                  n_seq_states
                                  )

    # This is done using Hidden Markov Models but since 
    # Transition matrix is uniformly distributed we can use random sequency
    #     mc = mcmix(nBS,'Fix',ones(nBS)*(1/nBS));
    #     seqBS= simulate(mc,nseqBS-1);
    seqBS = np.random.randint(0, n_brain_states, n_seq_states)

    full_bs_matrix = np.zeros((n_brain_states, n_nodes, n_nodes, model.order))

    for j in range(n_brain_states):
        matrix = np.eye(n_nodes)
        selected_edges = selected_states[j]
        for edge in selected_edges:
            matrix[edges[np.array(edge)]] = 1
        pl.figure()
        pl.imshow(matrix)
        matrix = np.dstack([matrix for _ in range(model.order)])

        cycles = 0
        FA = np.zeros((n_nodes*model.order, n_nodes*model.order))
        eye_idx = n_nodes*model.order - n_nodes
        FA[n_nodes:, :eye_idx] = np.eye(eye_idx)
        for k in range(10000):
            A = 1/2.5 * np.random.rand(n_nodes, n_nodes, model.order) * matrix
            FA[:n_nodes,:] = np.reshape(A, (n_nodes, -1), 'F')

            eig, _ = sp.linalg.eig(FA)

            if np.all(np.abs(eig) < 1):
                print(k)
                break

        
        if k==10000:
            raise("Solutions not found")



        full_bs_matrix[j] = A.copy()


    data = model.fit(full_bs_matrix, seqBS, length_bs)


           
    """    
    % writing the structure with the useful info
    structure.whichBS{iiter}=whichBS;
    structure.seqBS{iiter}=seqBS;
    structure.lengthBS{iiter}=lengthBS;
    structure.probtrans{iiter}=mc.P;
    structure.data{iiter}=data;
    
    SNR = 10;
    noise = randn(size(data));
    datanoisy = data + sqrt(1/SNR)*(noise./std(noise)); 
    structure.SNR = SNR;
    structure.datanoisy{iiter} = datanoisy;       
    """
data = data.T
window_length = 1 * sample_frequency
for i in range(n_iteration):
    
    # Butter filter
    """
    iir_params = dict(order=8, ftype='butter')
    filt = mne.filter.create_filter(data, sample_frequency, l_freq=6, h_freq=20,
                                    method='iir', iir_params=iir_params,
                                    verbose=True)

    data_filtered = signal.sosfiltfilt(filt['sos'], data.T).T
    """
    #b, a = signal.butter(8, [6, 20], btype='bandpass', fs=sample_frequency)
    #data_filtered = signal.filtfilt(b, a, data.T).T

    # Window-length
    connectivity_lenght = data.shape[1] - window_length + 1
    timewise_connectivity = np.zeros((n_edges, connectivity_lenght))
    for w in range(connectivity_lenght):
        data_window = data[:, w:w+window_length]
        assert data_window.shape[1] == 256
        phi = np.angle(signal.hilbert(data_window))
        for e, (x, y) in enumerate(edges):
            coh = np.imag(np.exp(1j*(phi[x] - phi[y])))
            iplv = np.abs(np.mean(coh))
            timewise_connectivity[e, w] = iplv

        print(w/connectivity_lenght*100)

    # Storing structure n_edges x (time - window)
    b, a = signal.butter(4, 2*(1/window_length), btype='lowpass', fs=sample_frequency)
    iplv_filtered = signal.filtfilt(b, a, timewise_connectivity.T).T





stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.';

conn = structure.cts{1};
nsec = 60;
n = 256*nsec; figure; plot([1:n]/256,[conn(:,1:n); stat(1:n).'/5] )

% save the structure
% save('newstruct-16122019_AR.mat','structure','-v7.3')

%% K means clustering % FIXME: ciclo sulle ripetizioni!!!!

% FIXME: data-driven identification of number of clusters (Roberto's algorithms)

idx = kmeans(structure.cts_filt{1}.',6);

stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.'+1;

% find the right estimated state order
UNCI = unique(idx); % uniquely numbered cluster indexes
P = perms(UNCI);
ERR = zeros(size(UNCI,1),1);
for i=1:size(P,1) % for all permutations
    idx_loc = idx;
    for j=1:length(UNCI)
        idx_loc(idx==UNCI(j)) = P(i,j);
    end
    ERR(i)=sum(double( not(idx_loc==stat) ))/numel(idx_loc);
end

% choose the permutation which gives the minimum ERROR
[v,imin] = min(ERR);
idx_loc = idx;
for j=1:length(UNCI)
   idx_loc(idx==UNCI(j)) = P(imin,j);
end
    
figure
nsec = 120;
n = 256*nsec; figure; plot([1:n]/256,[idx_loc(1:n)'; stat(1:n)'] )





%% plots

% example of BSs considered
figure('Position', [100 100 900 900])
for ibs=1:nBS
    subplot(floor(nBS/2),2,ibs)
    WBS=whichBS(ibs,isnan(whichBS(ibs,:))==0);
    G = digraph(1,1);
    for inode=2:nNodes
        G = addedge(G,inode,inode);
    end
    if(length(WBS)>0)
      for il=1:length(WBS)
          G = addedge(G,Edges(WBS(il),1),Edges(WBS(il),2));
      end
    end
    plot(G,'Layout','force')
    axis off
    str = sprintf('Brain state %d', ibs);
    title(str)
end

% graph of the BSs and transition probability matrix
figure('Position', [100 100 900 900])
subplot(2,1,1)
graphplot(mc,'ColorEdges',true);
colormap parula
caxis([0 0.3])
subplot(2,1,2)
imagesc(mc.P);
xlabel('Brain state')
ylabel('Brain state')
title('Transition probability')
colormap parula
caxis([0 0.3])
colorbar

% example of BSs sequence 
figure('Position', [100 100 900 400])
seqplot=[];
for iBS=1:length(lengthBS)
    seqplot=[seqplot,seqBS(iBS)*ones(1,lengthBS(iBS))];
end
plot(1/256:1/256:60,seqplot(1:256*60))
ylabel('Brain state')
xlabel('Sec')

% example of time series realization
figure('Position', [100 100 900 900])
for iplot=1:nNodes
    subplot(nNodes,1,iplot)
    plot(1/256:1/256:60,data(1:256*60,iplot))
    ylabel('Simulated time series')
    xlabel('Sec')
end

% histogram of life time for each BS
figure('Position', [100 100 900 400])
hist(seqBS,10)
ylabel('Occurrency')
xlabel('Brain state')
