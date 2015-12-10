% Use PCA

load('../../data/provideIdx.mat');
load('../../data/missIdx.mat');
load('../../data/provideData_1000.mat');
% load('../../data/events_1000.mat');
load('../../data/Train.mat');
load('../../data/Test.mat');
NVOXEL = 5903; % total number of voxels
NMISSING = size(missIdx,2);
NDATA = size(provideData,1);

pred = zeros(1000, 2731); % result prediction to write

%% Do prediction here: using PCA
CUTOFF = 50;
K = 20;

% project train data onto pca
completeData = vertcat(Xtrain, Xtest);
completeDataS = completeData(:,provideIdx);
xmean = mean(completeDataS,1);
[coeff,score,latent] = pca(completeDataS);
completeDataSPCA = score(:,1:CUTOFF);

% project test data onto pca
missingDataPCA = (provideData-repmat(xmean, size(provideData,1), 1)) * coeff(:,1:CUTOFF);

% calculate distance and do prediction
dist = pdist2(missingDataPCA,completeDataSPCA);
[dist, idx] = sort(dist, 2, 'ascend');
for i = 1:size(pred,1)
    for j = 1:K
        pred(i,:) = pred(i,:)+completeData(idx(i,j), missIdx);
    end
end
pred = pred/K;
%%
csvwrite('prediction.csv',pred); % remember to compress to zip before turning into Autolab!
% zip prediction.csv.zip prediction.csv