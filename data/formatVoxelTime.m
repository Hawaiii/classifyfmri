% load('Test.mat');
% load('Train.mat');

% X = [];
% Y = [];
% 
% for i = 1:size(Xtrain,2)
%     Xp = [repmat([x(i) y(i) z(i)], size(Xtrain,1),1) eventsTrain];
%     X = vertcat(X, Xp);
%     
%     Y = vertcat(Y, Xtrain(:,i));
%     if mod(i, 1000) == 0
%         i
%     end
% end

for i = 1:size(Xtest,2)
    Xp = [repmat([x(i) y(i) z(i)], size(Xtest,1),1) eventsTest];
    X = vertcat(X, Xp);
    
    Y = vertcat(Y, Xtest(:,i));
end

save('voxelTimeTrain.mat', 'X', 'Y');
