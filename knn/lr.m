% Each row of Xtrain is a data point 501 x 5903, of 27 subjects total
% Ytrain: 0 - no event; 1 - early stop; 3 - correct go
% eventsTrain: divde into 1:351, and 352:702 two scanning runs
% x,y,z: 3D coordinate of fMRI scans

% 20% - valid csv
% (20+30)% - baseline svm
% TODO 100% - 66.7 accurancy
% TODO extracredit
% TODO normalize data 
format shortg
load('../data/Train.mat');
load('../data/Test.mat');
nClass = size(unique(Ytrain),1);
nTest = size(Xtest,1);

%% TODO Train baseline
% TODO Set the hinge weight C=50 in the objective wTw/2+C?ih(yixi?w).

% Trying PCA
[coeff,score,latent] = pca(Xtrain);
XtestOrig = Xtest;
CUTOFF1 = 90
CUTOFF2 = 90
CUTOFF3 = 90
totalcverr=0;
% m = 1;
% K=10
%% functionname: function description



Xtrain = score(:,1:CUTOFF1);
Xtest = XtestOrig * coeff(:,1:CUTOFF1);
Yorig = Ytrain;
for i=1:size(Xtrain,1)
	if Yorig(i)==3;
		Ytrain(i)=0;
	elseif Yorig(i)==1;
		Ytrain(i)=0;
	else
		Ytrain(i)=1;
	end
end


model = glmfit(Xtrain, Ytrain, 'binomial');
totalcverr = totalcverr+ calcverr(Xtrain,Ytrain);
% The glmval function evaluates a given model for a set of data with a
% given linking function
y_hat(1,:) = glmval(model, Xtest, 'logit');


Xtrain = score(:,1:CUTOFF2);
Xtest = XtestOrig * coeff(:,1:CUTOFF2);
Ytrain = Yorig;
for i=1:size(Xtrain,1)
	if Yorig(i)==3;
		Ytrain(i)=0;
	end
end


model = glmfit(Xtrain, Ytrain, 'binomial');
totalcverr = totalcverr+calcverr(Xtrain,Ytrain);
% The glmval function evaluates a given model for a set of data with a
% given linking function
y_hat(2,:) = glmval(model, Xtest, 'logit');



Xtrain = score(:,1:CUTOFF3);
Xtest = XtestOrig * coeff(:,1:CUTOFF3);
Ytrain = Yorig;
for i=1:size(Xtrain,1)
	if Yorig(i)==3;
		Ytrain(i)=1;
	elseif Yorig(i)==1;
		Ytrain(i)=0;
	end
end


model = glmfit(Xtrain, Ytrain, 'binomial');

% The glmval function evaluates a given model for a set of data with a
% given linking function
totalcverr = totalcverr+calcverr(Xtrain,Ytrain);
y_hat(3,:) = glmval(model, Xtest, 'logit');


% Ytrain
% model = mnrfit(Xtrain, Ytrain);
% pred = mnrval(model, Xtrain, 'logit')
% y_hat = transpose(y_hat)
% y_max = transpose(sum(y_hat))
% pred = transpose(y_hat);
% y_max = repmat(y_max,1,3);
% pred = pred./y_max
% pred = round( pred, 1); 
% pred(:, 3) = 1 - sum(pred(:, 1:2), 2);
[M,I] = max(y_hat);
csvwrite('y_hat.csv',transpose(y_hat));
csvwrite('max_y_hat.csv',transpose(M));
I;
y_hat;


pred = zeros(nTest, nClass);
load('pred.mat');
for i=1:nTest
	if ( M(i)>0.45&&M(i)<0.55)
		
		k = find(pred_svm(i,:));
		if(size(k,2)==0)
			k = find(pred_knn(i,:));
			k
		end
		pred(i,k)=1;
	else
		pred(i,I(i))=1;
	end
end

pred;
totalcverr/3
% pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
% pred((pred01==0) & (pred03 ==0), 1) = 1;
% pred((pred01==0) & (pred03 ==0), 2:3) = 0;
% pred((pred01==1) & (pred13 ==1), 1) = 0;
% pred((pred01==1) & (pred13 ==1), 2) = 1;
% pred((pred01==1) & (pred13 ==1), 3) = 0;
% pred((pred03==3) & (pred13 ==3), 1:2) = 0;
% pred((pred03==3) & (pred13 ==3), 3) = 1;

% Write prediction.csv
csvwrite('prediction.csv',pred);