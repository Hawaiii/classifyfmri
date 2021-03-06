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
nTrain = size(Xtrain,1);

%% TODO Train baseline
% TODO Set the hinge weight C=50 in the objective wTw/2+C?ih(yixi?w).

% Trying PCA
CUTOFF = 500;
[coeff,score,latent] = pca(Xtrain);
whos('coeff')
whos('score')
whos('latent')
Xtrain = score(:,1:CUTOFF);
subjects = unique(subjectsTrain(:, 1))
for i = 1:size(subjects,1)
	subID = subjects(i)
	subID
   	SubClassifier{subID} = fitcnb(Xtrain(subjectsTrain(:, 1) == subID, :),Ytrain(subjectsTrain(:, 1) == subID, 1));
end

SubClassifier
% Subject(i) = Xtrain(SubjectsTrain(:, 1) == i, :);

% mdl = fitcnb(Xtrain,Ytrain);
% mdl
% rloss = resubLoss(mdl)
% cvmdl = crossval(mdl);
% kloss = kfoldLoss(cvmdl)
Xtest = Xtest * coeff(:,1:CUTOFF);
for i=1:nTest
	i
	pred013(i) = predict(SubClassifier(subjectsTest(i,1)),Xtest(i,:))
end
% pred013 = predict(mdl,Xtest);
pred = zeros(nTest, nClass);
% pred(:,1) = 1;
pred((pred013==0),1) = 1;
pred((pred013==0),2:3) = 0;
pred((pred013==1),2) = 1;
pred((pred013==1),1) = 0;
pred((pred013==1),3) = 0;
pred((pred013==3),3) = 1;
pred((pred013==3),1:2) = 0;
pred;
csvwrite('prediction.csv',pred)
% pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
% pred((pred01==0), 1) = 1;
% pred((pred01==0) & (pred03 ==0), 2:3) = 0;
% pred((pred01==1) & (pred13 ==1), 1) = 0;
% pred((pred01==1) & (pred13 ==1), 2) = 1;
% pred((pred01==1) & (pred13 ==1), 3) = 0;
% pred((pred03==3) & (pred13 ==3), 1:2) = 0;
% pred((pred03==3) & (pred13 ==3), 3) = 1;

