% for each subject
% calculate the brain activity for each time

% subjectID = 6; %1-4,6-28
label = 1; % 0, 1, 3
load('../data/Train.mat');
load('../data/Test.mat');
load('../data/YTest.mat');

trainSelect = Ytrain == label;
testSelect = Ytest == label;
X = vertcat(Xtrain(trainSelect,:), Xtest(testSelect,:));
time = vertcat(eventsTrain(trainSelect,:), eventsTest(testSelect,:));


[time, idx] = sort(time);
ordX = X(idx,:);
numscan = size(time,1);

change = zeros(max(time),size(ordX,2));
for i = 2:numscan
    tmp = abs((ordX(i,:)-ordX(i-1,:))/(time(i)-time(i-1)));
    for j = time(i-1):time(i)
        change(j,:) = tmp;
    end
end

% smoothing it
gaussFilter = gausswin(31);
gaussFilter = gaussFilter / sum(gaussFilter); % Normalize.

% Do the blur.
changesum = sum(change,2);
ssum = conv(changesum, gaussFilter)

plot(ssum);
xlabel('time')
ylabel('change')
