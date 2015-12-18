% plot brain change across time for one subject

subjectID = 1; %1-4,6-28
load('../data/Train.mat');
load('../data/Test.mat');

trainSelect = subjectsTrain == subjectID;
testSelect = subjectsTest == subjectID;
X = vertcat(Xtrain(trainSelect,:), Xtest(testSelect,:));
normX = (X-min(X(:)))/(max(X(:))-min(X(:)));
normX(isnan(normX)) = 0;
time = vertcat(eventsTrain(trainSelect,:), eventsTest(testSelect,:));

[time, idx] = sort(time);
ordnormX = normX(idx,:);
numscan = size(time,1);
figure()
for i = 1:numscan
    scatter3(x,y,z,[],ordnormX(idx(i),:)','filled');
    if i > 1 pause(0.1*(time(i)-time(i-1))); end
end