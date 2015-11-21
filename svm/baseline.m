% Each row of Xtrain is a data point 501 x 5903, of 27 subjects total
% Ytrain: 0 - no event; 1 - early stop; 3 - correct go
% eventsTrain: divde into 1:351, and 352:702 two scanning runs
% x,y,z: 3D coordinate of fMRI scans

% TODO 20% - valid csv
% TODO (20+30)% - baseline svm
% TODO 100% - 66.7 accurancy
% TODO extracredit

load('../data/Train.mat');
% load('../data/Test.mat');
nClass = size(unique(YTrain),1);
nTest = size(XTest,1);

% TODO Train baseline

% From the inputs, use only X (the voxel array).
% Use a Gaussian RBF kernel, i.e., k(x,x?)=exp(??x?x??/2?2).
% Set kernel width ?=100.
% Set the hinge weight C=50 in the objective wTw/2+C?ih(yixi?w).
% Fit three binary 1-vs-1 classifiers (one for each pair of classes).
% When testing, use majority vote among the three classifiers: e.g., if 
% the 0-1 classifier says 1, the 0-3 classifier says 3, and the 1-3 
% classifier says 1, predict 1.

% TODO Test
pred = zeros(nTest, nClass);

% TODO Write prediction.csv