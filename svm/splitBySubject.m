% Reformat the training data by subject number
load('../data/Train.mat');
nSubject = max(subjectsTrain(:));
XSubtrain = cell(1, nSubject);
for i = 1:nSubject
    idx = (subjectsTrain==i);
    subi = struct('subID',i,'X', Xtrain(idx,:),'Y',Ytrain(idx,:));
    XSubtrain{1,i} = subi;
end

avgY = 0;
countS = 0;
for i = 1:nSubject
    avgY = avgY + size(unique(XSubtrain{1,i}.Y),1);
end
avgY/27

save('XSubtrain.mat','XSubtrain');