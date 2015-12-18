% for each subject
% calculate the brain activity for each time

subjectID = 6; %1-4,6-28
load('../data/Train.mat');
load('../data/Test.mat');

trainSelect = subjectsTrain == subjectID;
testSelect = subjectsTest == subjectID;
X = vertcat(Xtrain(trainSelect,:), Xtest(testSelect,:));
time = vertcat(eventsTrain(trainSelect,:), eventsTest(testSelect,:));

[time, idx] = sort(time);
ordX = X(idx,:);
numscan = size(time,1);

change = zeros(max(time),size(ordX,2));
for i = 2:numscan
    tmp = (ordX(i,:)-ordX(i-1,:))/(time(i)-time(i-1));
    for j = time(i-1):time(i)
        change(j,:) = tmp;
    end
end

changesum = sum(change,2);
plot(changesum);
