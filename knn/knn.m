% Each row of Xtrain is a data point 501 x 5903, of 27 subjects total
% Ytrain: 0 - no event; 1 - early stop; 3 - correct go
% eventsTrain: divde into 1:351, and 352:702 two scanning runs
% x,y,z: 3D coordinate of fMRI scans

% 20% - valid csv
% (20+30)% - baseline svm
% TODO 100% - 66.7 accurancy
% TODO extracredit

load('../data/Train.mat');
load('../data/Test.mat');
nClass = size(unique(Ytrain),1);
nTest = size(Xtest,1);
nSubject = max(subjectsTrain(:));

%% Train

% Trying PCA

% CUTOFF = 200;
% [coeff,score,latent] = pca(Xtrain);
% Xtrain = score(:,1:CUTOFF);
% prefix = 'subjectID'
% subjects = unique(subjectsTrain)
% for i = 1:size(subjects,1)
% 	subID = subjects(i);
% 	% subID
% 	% strcat(prefix,int2str(subID))
% 	assignin('base',strcat(prefix,int2str(subID)),fitcknn(Xtrain(subjectsTrain(:, 1) == subID, :),Ytrain(subjectsTrain(:, 1) == subID, 1),'NumNeighbors',3));
% end

% SubClassifier
minkloss = 100000000;

for i=1:10
	mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',i,'NSMethod','exhaustive','Distance','correlation');
	% mdl
	% rloss = resubLoss(mdl)
	cvmdl = crossval(mdl);
	kloss = kfoldLoss(cvmdl)
	if kloss<minkloss
		minmdl = mdl;
		minkloss = kloss;
	end
end
mdl = minmdl
% Xtest = Xtest * coeff(:,1:CUTOFF);
% for i=1:nTest
	
% 	testID = num2str(subjectsTest(i,1));
% 	classifier = eval(strcat(prefix, testID));
% 	% throwing error
% 	% 	Error using predict (line 84)
% 	% Systems of cell class cannot be used with the "predict" command. Convert the system to an identified model first, such as by using the "idss"
% 	% command.

% 	% Error in knn (line 39)
% 	% 	pred013(i) = predict(classifier,Xtest(i,:))
% 	pred013(i) = predict(classifier,Xtest(i,:));
% end
pred013 = predict(mdl,Xtest);

pred = zeros(nTest, nClass);
pred((pred013==0),1) = 1;
pred((pred013==1),2) = 1;
pred((pred013==3),3) = 1;

pred((pred013==3),1:2) = 0;
pred;
pred_knn =pred;
% Write prediction.csv
save('pred.mat','pred_knn');
csvwrite('prediction.csv',pred)

