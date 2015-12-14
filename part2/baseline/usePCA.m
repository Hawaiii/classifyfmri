% Use PCA

load('../../data/provideIdx.mat');
load('../../data/missIdx.mat');
% load('../../data/provideData_1000.mat');
% load('../../data/events_1000.mat');
load('../../data/Train.mat');
load('../../data/Test.mat');
NVOXEL = 5903; % total number of voxels
NMISSING = size(missIdx,2);
NDATA = size(provideData,1);


%% Do prediction here: using PCA
rmse = zeros(16,3); rmseptr = 1;
testRatio = 0.1;
for CUTOFF = 10: 50:200
    for K = 1:5:20
        % project train data onto pca
        fullData = vertcat(Xtrain, Xtest);

        trainidx = randperm(size(fullData, 1));
        trainCutoff = round((1-testRatio)*size(fullData,1));
        trainFullData = fullData(trainidx(1:trainCutoff),:);
        testFullData = fullData(trainidx(trainCutoff+1:end),:);

        pred = zeros(size(testFullData,1), 2731); % result prediction to write
        
        fullDataS = trainFullData(:,provideIdx);
        xmean = mean(fullDataS,1);
        [coeff,score,latent] = pca(fullDataS);
        fullDataSPCA = score(:,1:CUTOFF);

        % project test data onto pca
        provideData = testFullData(:,provideIdx);
        missingDataPCA = (provideData-repmat(xmean, size(provideData,1), 1)) * coeff(:,1:CUTOFF);

        % calculate distance and do prediction
        dist = pdist2(missingDataPCA,fullDataSPCA);
        [dist, idx] = sort(dist, 2, 'ascend');
        for i = 1:size(pred,1)
            for j = 1:K
                pred(i,:) = pred(i,:)+fullData(idx(i,j), missIdx);
            end
        end
        pred = pred/K;

        % calculate RMSE
        rmse(rmseptr,1) = CUTOFF;
        rmse(rmseptr,2) = K;
        rmse(rmseptr,3) = sqrt( sum(sum( (testFullData(:,missIdx)-pred).^2)) / numel(pred) );
        
        rmseptr = rmseptr+1;
    end
end

figure;
scatter3(rmse(:,1), rmse(:,2), rmse(:,3))
xlabel('cutoff')
ylabel('k')
title('rmse')

%%
% csvwrite('prediction.csv',pred); % remember to compress to zip before turning into Autolab!
% zip prediction.csv.zip prediction.csv