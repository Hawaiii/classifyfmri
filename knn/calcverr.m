function [cverr] = calcverr(Xtrain,Ytrain)
	num_shuffles =10;
    num_folds = 5;
    err=0;
	totalPred=0;
	for j = 1:num_shuffles
    indices = crossvalind('Kfold',Ytrain,num_folds);
    for i = 1:num_folds
        test = (indices == i); train = ~test;
        [b,dev,stats] = glmfit(Xtrain(train,:),Ytrain(train),'binomial','logit'); % Logistic regression
        size(Ytrain(test));
        size(Xtrain(test,:));
        err = err + nnz((Ytrain(test) - glmval(b,Xtrain(test,:),'logit'))>0.5);
        totalPred = totalPred + size(Ytrain(test),1);
    end
end
cverr = err/totalPred;