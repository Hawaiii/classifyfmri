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

%% TODO Train baseline
% TODO Set the hinge weight C=50 in the objective wTw/2+C?ih(yixi?w).

% Trying PCA
CUTOFF = 150;
[coeff,score,latent] = pca(Xtrain);
Xtrain = score(:,1:CUTOFF);

select01 = (Ytrain==0)|(Ytrain==1);
svm01 = svmtrain(Xtrain(select01,:),Ytrain(select01), ...
    'kernel_function', 'rbf','rbf_sigma',100);
select03 = (Ytrain==0)|(Ytrain==3);
svm03 = svmtrain(Xtrain(select03,:),Ytrain(select03), ...
    'kernel_function', 'rbf','rbf_sigma',100);
select13 = (Ytrain==1)|(Ytrain==3);
svm13 = svmtrain(Xtrain(select13,:),Ytrain(select13), ...
    'kernel_function', 'rbf','rbf_sigma',100);

%% Test
% When testing, use majority vote among the three classifiers: e.g., if 
% the 0-1 classifier says 1, the 0-3 classifier says 3, and the 1-3 
% classifier says 1, predict 1.

% Try PCA
Xtest = Xtest * coeff(:,1:CUTOFF);

pred01 = svmclassify(svm01, Xtest);
pred03 = svmclassify(svm03, Xtest);
pred13 = svmclassify(svm13, Xtest);

pred = zeros(nTest, nClass);
pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
pred((pred01==0) & (pred03 ==0), 1) = 1;
pred((pred01==0) & (pred03 ==0), 2:3) = 0;
pred((pred01==1) & (pred13 ==1), 1) = 0;
pred((pred01==1) & (pred13 ==1), 2) = 1;
pred((pred01==1) & (pred13 ==1), 3) = 0;
pred((pred03==3) & (pred13 ==3), 1:2) = 0;
pred((pred03==3) & (pred13 ==3), 3) = 1;

% Write prediction.csv
csvwrite('prediction.csv',pred);