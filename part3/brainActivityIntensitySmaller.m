% calculate the brain activity for each time

load('../data/Train.mat')
load('../data/Test.mat')
load('../data/Ytest.mat')

X_new = csvread('X_new_1.csv');
label = 1;

trainSelect = Ytrain == label;
testSelect = Ytest == label;
X = X_new(vertcat(trainSelect,testSelect),:);
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

changesum = sum(change,2);
plot(changesum);
