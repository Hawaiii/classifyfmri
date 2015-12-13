load('Test.mat');
load('Train.mat');
load('voxelTimeTrain2.mat');

X3 = [];
Y3 = [];
% 
for i = 1:size(Xtest,2)
    Xp = [repmat([x(i) y(i) z(i)], size(Xtest,1),1) eventsTest];
    X3 = vertcat(X3, Xp);
    Y3 = vertcat(Y3, Xtest(:,i));
end
X3 = vertcat(X3, X);
Y3 = vertcat(Y3, Y);
save('../../data/voxelTimeTrainTest.mat', 'X3', 'Y3');
% X2=[]
% Y2=[]
% for i = 1:size(provideData,2)
%     idx = provideIdx(i);
%     Xp = [repmat([x(idx) y(idx) z(idx)], size(provideData,1),1) transpose(events)];
%     X2 = vertcat(X2, Xp);
    
%     Y2 = vertcat(Y2, provideData(:,i));
% end

% save('../../data/voxelTimeTrain2.mat', 'X2', 'Y2');
