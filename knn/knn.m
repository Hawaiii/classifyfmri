% Each row of Xtrain is a data point 501 x 5903, of 27 subjects total
% Ytrain: 0 - no event; 1 - early stop; 3 - correct go
% eventsTrain: divde into 1:351, and 352:702 two scanning runs
% x,y,z: 3D coordinate of fMRI scans

% 20% - valid csv
% (20+30)% - baseline svm
% TODO 100% - 66.7 accurancy
% TODO extracredit

load('../data/Train.mat');
load('../data/Test.mat');
nClass = size(unique(Ytrain),1);
nTest = size(Xtest,1);
nSubject = max(subjectsTrain(:));

%% Train

% Trying PCA
CUTOFF = 50;
[coeff,score,latent] = pca(Xtrain);
trainavg = mean(Xtrain,1);
Xtrain = score(:,1:CUTOFF);
SubClassifier = cell(1, nSubject); %An cell array
for subID = 1:nSubject
    idx = (subjectsTrain==subID);
    if sum(idx) > 0
       	SubClassifier{1,subID} = fitcknn(Xtrain(idx, :),Ytrain(idx, 1),'NumNeighbors',3);
    end
end

% SubClassifier
% mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',1);
% mdl
% rloss = resubLoss(mdl)
% cvmdl = crossval(mdl);
% kloss = kfoldLoss(cvmdl)
Xtest = (Xtest-repmat(trainavg,size(Xtest,1),1)) * coeff(:,1:CUTOFF);
pred013 = zeros(nTest,1);
for i=1:nTest
	classifier = SubClassifier{subjectsTest(i,1)};
	pred013(i,1) = predict(classifier,Xtest(i,:));
end
pred = zeros(nTest, nClass);
pred((pred013==0),1) = 1;
pred((pred013==1),2) = 1;
pred((pred013==3),3) = 1;
csvwrite('prediction.csv',pred);
