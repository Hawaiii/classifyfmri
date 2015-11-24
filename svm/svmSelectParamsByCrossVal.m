function svmSelectParamsByCrossVal
load('../data/Train.mat');
nClass = size(unique(Ytrain),1);
nTrain = size(Xtrain,1);

PCACUTOFF = 10:50:500;
SVMKERNELFUN = {'linear','quadratic','polynomial','rbf','mlp'};
SVMRBFSIG = 50:50:200;
SVMBOXC = 1:10:100;
% TODO: should also try learn svm on diff subject with XSubtrain

bestParams = struct([]);
Params = {};
ParamsIdx = 1;
bestPerformance = 0;

[coeff,score,latent] = pca(Xtrain);
trainCenter = mean(Xtrain,1);

svmoption = optimset('maxiter',15000);

for pcacutoff = PCACUTOFF
    Xtrain = score(:,1:pcacutoff);

% Partition into 90 percent train and 10 percent validation
    for i = 1:10
        for svmboxc = SVMBOXC,
            for svmkernelfun = SVMKERNELFUN,
                svmkernelfun = svmkernelfun{1};
                [trainNum,valNum,~] = dividerand(nTrain,0.9,0.1,0);
                trainInd = false(nTrain,1);
                trainInd(trainNum,1) = 1;
                valInd = false(nTrain,1);
                valInd(valNum,1) = 1;
                nTest = sum(valInd);

                if strcmp(svmkernelfun,'rbf')
                    for svmrbfsig = SVMRBFSIG
                        try 
                        select01 = trainInd & ((Ytrain==0)|(Ytrain==1));
                        svm01 = svmtrain(Xtrain(select01,:),Ytrain(select01), ...
                        'kernel_function', svmkernelfun,'rbf_sigma',svmrbfsig,'boxconstraint',svmboxc,'options',svmoption);
                        select03 = trainInd & ((Ytrain==0)|(Ytrain==3));
                        svm03 = svmtrain(Xtrain(select03,:),Ytrain(select03), ...
                        'kernel_function', svmkernelfun,'rbf_sigma',svmrbfsig,'boxconstraint',svmboxc,'options',svmoption);
                        select13 = trainInd & ((Ytrain==1)|(Ytrain==3));
                        svm13 = svmtrain(Xtrain(select13,:),Ytrain(select13), ...
                        'kernel_function', svmkernelfun,'rbf_sigma',svmrbfsig,'boxconstraint',svmboxc,'options',svmoption);
                    
                        % Test
                        Xtest = Xtrain(valInd,:);

                        pred01 = svmclassify(svm01, Xtest);
                        pred03 = svmclassify(svm03, Xtest);
                        pred13 = svmclassify(svm13, Xtest);

                        pred = zeros(nTest, nClass);
                        pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
                        pred((pred01==0) & (pred03 ==0), 1) = 1;
                        pred((pred01==0) & (pred03 ==0), 2:3) = 0;
                        pred((pred01==1) & (pred13 ==1), 1) = 0;
                        pred((pred01==1) & (pred13 ==1), 2) = 1;
                        pred((pred01==1) & (pred13 ==1), 3) = 0;
                        pred((pred03==3) & (pred13 ==3), 1:2) = 0;
                        pred((pred03==3) & (pred13 ==3), 3) = 1;
                        
                        accurancy = calcAccuracy(valInd, pred, Ytrain);
                        Params{1,ParamsIdx} = struct(['PCACUTOFF',pcacutoff,...
                                'SVMKERNELFUN', svmkernelfun,...
                                'SVMRBFSIG',svmrbfsig,'SVMBOXC',svmboxc,...
                                'performance', accurancy]);
                        if bestPerformance < accurancy
                            bestPerformance = accurancy;
                            bestParams = Params{1,ParamsIdx};
                        end
                        ParamsIdx = ParamsIdx+1;
                        catch
                            1
                        end
                    end
                    
                else
                try
                select01 = trainInd & ((Ytrain==0)|(Ytrain==1));
                svm01 = svmtrain(Xtrain(select01,:),Ytrain(select01), ...
                'kernel_function', svmkernelfun,'boxconstraint',svmboxc,'options',svmoption);
                select03 = trainInd & ((Ytrain==0)|(Ytrain==3));
                svm03 = svmtrain(Xtrain(select03,:),Ytrain(select03), ...
                'kernel_function', svmkernelfun,'boxconstraint',svmboxc,'options',svmoption);
                select13 = trainInd & ((Ytrain==1)|(Ytrain==3));
                svm13 = svmtrain(Xtrain(select13,:),Ytrain(select13), ...
                'kernel_function', svmkernelfun,'boxconstraint',svmboxc,'options',svmoption);
            
                % Test
                Xtest = Xtrain(valInd,:);

                pred01 = svmclassify(svm01, Xtest);
                pred03 = svmclassify(svm03, Xtest);
                pred13 = svmclassify(svm13, Xtest);

                pred = zeros(nTest, nClass);
                pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
                pred((pred01==0) & (pred03 ==0), 1) = 1;
                pred((pred01==0) & (pred03 ==0), 2:3) = 0;
                pred((pred01==1) & (pred13 ==1), 1) = 0;
                pred((pred01==1) & (pred13 ==1), 2) = 1;
                pred((pred01==1) & (pred13 ==1), 3) = 0;
                pred((pred03==3) & (pred13 ==3), 1:2) = 0;
                pred((pred03==3) & (pred13 ==3), 3) = 1;

                accurancy = calcAccuracy(valInd, pred, Ytrain);
                Params{1,ParamsIdx} = struct('PCACUTOFF',pcacutoff,...
                        'SVMKERNELFUN', svmkernelfun,...
                        'SVMRBFSIG',0,'SVMBOXC',svmboxc,...
                        'performance', accurancy);
                if bestPerformance < accurancy
                    bestPerformance = accurancy;
                    bestParams = Params{1,ParamsIdx};
                end
                ParamsIdx = ParamsIdx+1;
                catch
                    1
                end
                end
            end
        end
    end

end
end

function [accurancy] = calcAccuracy(idx, prediction, labels)
% Input:
%  idx: 501x1 logical array, where 1 selects the data points, sum(idx)=M.
%  (PREV) Mx1 array of indices, where 1 selects the data points.
%  prediction: Mx3 array, containing M predictions of selected data points,
%              first column denotes prob. of label 0, second 1, last 3.
%  labels: 501x1, Ytrain.

gt = labels(idx,:);
accurancy = 0;
accurancy = accurancy + sum(prediction(gt==0, 1));
accurancy = accurancy + sum(prediction(gt==1, 2));
accurancy = accurancy + sum(prediction(gt==3, 3));
end