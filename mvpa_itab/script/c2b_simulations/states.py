
class ConnectivtitySourceSimulator():
    def __init__(self, n_sorces=4, n_states=10, freq=256, n_changes=300):

        self._n_sources = n_sorces
        self._n_states = n_states
        self._freq = freq
        self._n_changes = n_changes
        self._avg_ts_lenght = freq * n_changes

import itertools

n_nodes = 4
n_states = 10
freq = 256
n_changes = 300
avg_ts_lenght = freq * n_changes

edges = [edge for edge in itertools.combinations(np.arange(n_nodes), 2)]
n_edges = len(edges)

brain_states = []
for i in range(n_edges):

    for j in itertools.combinations(np.arange(n_edges), i+1):

        brain_states.append([edges[e] for e in j])






% parameter settings
nNodes=4; % number of nodes
nBS=10; % number of brain states (BSs) to simulate 
nIter=1; % number of simulation realizations
order=5; % order of the MVAR model
fsample=256; %sampling frequency
Lt=60*fsample*5; % average length of the time series
nseqBS=Lt/fsample; % number of BS changes in one run/session

% selection of the brain states
Edges = nchoosek([1:nNodes],2); % possible edges in the graph
nEdges=(nNodes*(nNodes-1)/2); % total number of edges in the graph

% all the possible BSs
BS=NaN(1,nEdges);
k=1;
for iedge=1:nEdges
    combapp=nchoosek([1:nEdges],iedge);
    BS(k+1:k+size(combapp,1),1:nEdges) = [combapp,NaN(size(combapp,1),nEdges-iedge)];
    k=k+length(combapp);
end

% definition of the structure with useful parameters
structure.edges=Edges;
structure.possibleBS=BS;
structure.whichBS=cell(1,nIter);
structure.seqBS=cell(1,nIter);
structure.lengthBS=cell(1,nIter);
structure.probtrans=cell(1,nIter);
structure.data=cell(1,nIter);

% MVAR parameters shared by all the BSs
Mdl = varm(nNodes,order);
Mdl.Constant=zeros(nNodes,1);
Mdl.Covariance=eye(nNodes)*1/10;

ARmatrices=cell(1,nBS);

% for each realization...
for iiter=1:1
    iiter
    % which BSs (among the possible BSs) are we considering in this realization?
    whichBS=BS(randperm(size(BS,1),nBS),:);
    
    % sequence life times of the BSs
    lengthBS=randi([fsample/2,fsample*3/2],1,nseqBS);
    
    % generate Markov model and BSs sequence with random transition probabilities
    mc = mcmix(nBS);
    seqBS= simulate(mc,nseqBS-1);
    
    % generate lag matrices of the MVAR for each BS considered and check the stability
    for iBS=1:nBS
        matrix{iBS}=eye(nNodes);
        whichEdge=whichBS(iBS,isnan(whichBS(iBS,:)')==0);
        for iEdge=1:length(whichEdge)
            matrix{iBS}(Edges(whichEdge(iEdge),1),Edges(whichEdge(iEdge),2))=1;
        end
        %matrix{iBS}=matrix{iBS}+matrix{iBS}'-eye(size(matrix{iBS})); % to
        %make the interactions bidirectional
        unstable = true;
        numc2=0;
        while unstable && numc2<100000
            numc2 = numc2+1;
            A = 1/10*randn(nNodes,nNodes,order).*repmat(matrix{iBS},1,1,order);
            FA = [reshape(A,nNodes,nNodes*order);
                 eye(nNodes*(order-1),nNodes*(order-1)),zeros(nNodes*(order-1),nNodes)];
            if all(abs(eig(FA))<1); 
                unstable = false;    
            end
        end
        for ilag=1:order
            ARmatrices{iBS}{ilag}=squeeze(A(:,:,ilag));
        end
    end  
    
    % generate MVAR data according to the previously defined lag matrices
    data=[];  
    for iseq=1:nseqBS
       Mdl.AR=ARmatrices{seqBS(iseq)};
       dataBS = simulate(Mdl,lengthBS(iseq));
       dataBS=dataBS-mean(dataBS,1);
       data=[data;dataBS];
    end    
    
    % writing the structure with the useful info
    structure.whichBS{iiter}=whichBS;
    structure.seqBS{iiter}=seqBS;
    structure.lengthBS{iiter}=lengthBS;
    structure.probtrans{iiter}=mc.P;
    structure.data{iiter}=data;
       
end

% save the structure
save('data.mat','structure','-v7.3')

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