% Train 3 svms on each subject
load('XSubtrain.mat');
load('../data/train.mat', 'Ytrain');
nSubject = size(XSubtrain,2);

for i = 1:nSubject
    if size(XSubtrain{1,i}.X,1)>0
        Xtrain = XSubtrain{1,i}.X;
        Ytrain = XSubtrain{1,i}.Y;
        
        CUTOFF = min(100, size(Xtrain,1)-1);
        [coeff,score,latent] = pca(Xtrain);
        trainavg = mean(Xtrain,1);
        Xtrain = score(:,1:CUTOFF);
        XSubtrain{1,i} = setfield(XSubtrain{1,i},'trainavg',trainavg);
        XSubtrain{1,i} = setfield(XSubtrain{1,i},'coeff',coeff(:,1:CUTOFF));

        select01 = (Ytrain==0)|(Ytrain==1);
        svm01 = svmtrain(Xtrain(select01,:),Ytrain(select01), ...
            'kernel_function', 'rbf','rbf_sigma',100,'boxconstraint',50);
        select03 = (Ytrain==0)|(Ytrain==3);
        svm03 = svmtrain(Xtrain(select03,:),Ytrain(select03), ...
            'kernel_function', 'rbf','rbf_sigma',100,'boxconstraint',50);
        select13 = (Ytrain==1)|(Ytrain==3);
        svm13 = svmtrain(Xtrain(select13,:),Ytrain(select13), ...
            'kernel_function', 'rbf','rbf_sigma',100,'boxconstraint',50);
        XSubtrain{1,i} = setfield(XSubtrain{1,i}, 'svm01', svm01);
        XSubtrain{1,i} = setfield(XSubtrain{1,i}, 'svm03', svm03);
        XSubtrain{1,i} = setfield(XSubtrain{1,i}, 'svm13', svm13);       
    end
end

load('../data/test.mat', 'Xtest','subjectsTest');

% Test by looking up subject and using that
pred = ones(nTest, nClass)*0.33;
pred(:,1) = 0.34; % hack
for i = 1:nSubject
   testSubSelect = find(subjestsTest==i);
   XtestSub = Xtest(testSubSelect,:);
   if size(XtestSub,1)>0
       XtestSub = (XtestSub - repmat(XSubtrain{1,i}.trainavg, size(XtestSub,1),1))...
           *XSubtrain{1,i}.coeff;
       pred01 = svmclassify(svm01, Xtest);
       pred03 = svmclassify(svm03, Xtest);
       pred13 = svmclassify(svm13, Xtest);
       
       select0 = (pred01==0) & (pred03 ==0);
       pred(testSubSelect(select0), :) = [ones(size(select0,1),1) zeros(size(select0,1),2)];
       select1 = (pred01==1) & (pred13 ==1);
       pred(testSubSelect(select1),:) = [zeros(size(select1,1),1) ones(size(select1,1),1) zeros(size(select1,1),1)];
       select3 = (pred03==3) & (pred13==3);
       pred(testSubSelect(select3),:) = [zeros(size(select3,1),2) ones(size(select3,1),1)];
   end
end

csvwrite('prediction.csv',pred);