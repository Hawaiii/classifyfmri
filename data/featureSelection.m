load('Train.mat');
load('Test.mat');
nDim = size(Xtrain,2);

%% Smooth
SMOOTHBANDWIDTH = 9;
SMOOTHSIGMA = 2;
loc = [x y z];
weight = zeros(nDim, nDim);
for i = 1:nDim
    pos = loc(i,:);
    dist = pdist2(pos, loc);
    window = dist < SMOOTHBANDWIDTH;
    weight(i,:) = window.*(gaussmf(dist, [SMOOTHSIGMA 0]));
end

XtrainS = Xtrain*weight;
XtestS = Xtest*weight;

%% Voxel Discrimination Over 2 Time Chunk
NUMCHUNK = 2;
XtrainTS = zeros(size(XtrainS,1), nDim*NUMCHUNK);
firstScan = (eventsTrain <= 351);
XtrainTS(firstScan, 1:nDim) = XtrainS(firstScan, :);
secondScan = (eventsTrain >= 352);
XtrainTS(secondScan, nDim+1:end) = XtrainS(secondScan,:);

XtestTS = zeros(size(XtestS,1), nDim*NUMCHUNK);
firstScan = (eventsTest <= 351);
XtestTS(firstScan, 1:nDim) = XtestS(firstScan, :);
secondScan = (eventsTest >= 352);
XtestTS(secondScan, nDim+1:end) = XtestS(secondScan,:);

%% PCA
CUTOFF = 200;
[coeff,score,latent] = pca(XtrainTS);
trainCenter = mean(XtrainTS,1);
Xtrain = score(:,1:CUTOFF);

Xtest = (XtestTS-repmat(trainCenter, size(XtestTS,1),1))*coeff(:,1:CUTOFF);

save('FSTrain.mat','Xtrain','Ytrain','eventsTrain','subjectsTrain','x','y','z');
save('FSTest.mat','Xtest','eventsTest','subjectsTest');

