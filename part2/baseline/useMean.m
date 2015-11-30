% Use mean
% provideIdx: indices for voxels that are provided in the test examples 
%             A 1 x 3172 array of int, each in 1:5903, corresponding to
%             the voxels from the project part 1 data.
% missIdx: The voxels that you need to predict. 
%          A 1 x 2731 array of integer indices, and contains all the 
%          numbers in 1:5903 that are not in provideIdx.
% provideData_1000: Test examples. A 1000 x 3172 array of voxel values.  
%                   Each row is a partial fMRI scan.
% events_1000: The event timings for the 1000 fMRI scans. 
%              A 1 x 1000 matrix with the same interpretation as the 
%              event timings from project part 1.

load('../../data/provideIdx.mat');
load('../../data/missIdx.mat');
load('../../data/provideData_1000.mat');
load('../../data/events_1000.mat');
load('../../data/Train.mat');
NVOXEL = 5903; % total number of voxels
NMISSING = size(missIdx,2);
NDATA = size(provideData,1);

pred = zeros(1000, 2731); % result prediction to write

%% Do prediction here: using mean of all training data
xmean = mean(Xtrain,1);
pred = repmat(xmean(missIdx), NDATA,1);

%%
csvwrite('prediction.csv',pred); % remember to compress to zip before turning into Autolab!
% zip prediction.csv.zip prediction.csv