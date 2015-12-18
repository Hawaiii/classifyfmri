% for each subject
% calculate the brain activity for each time

subjectID = 1; %1-4,6-28
load('../data/Train.mat');
load('../data/Test.mat');

trainSelect = subjectsTrain == subjectID;
testSelect = subjectsTest == subjectID;
X = vertcat(Xtrain(trainSelect,:), Xtest(testSelect,:));
time = vertcat(eventsTrain(trainSelect,:), eventsTest(testSelect,:));

[time, idx] = sort(time);
ordX = X(idx,:);
numscan = size(time,1);
figure()
for i = 1:numscan
    scatter3(x,y,z,[],ordnormX(idx(i),:)','filled');
    if i > 1 pause(0.1*(time(i)-time(i-1))); end
end