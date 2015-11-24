% Each row of Xtrain is a data point 501 x 5903, of 27 subjects total
% Ytrain: 0 - no event; 1 - early stop; 3 - correct go
% eventsTrain: divde into 1:351, and 352:702 two scanning runs
% x,y,z: 3D coordinate of fMRI scans

% 20% - valid csv
% (20+30)% - baseline svm
% TODO 100% - 66.7 accurancy
% TODO extracredit
% TODO normalize data 
load('../data/Train.mat');
load('../data/Test.mat');
nClass = size(unique(Ytrain),1);
nTest = size(Xtest,1);

%% TODO Train baseline
% TODO Set the hinge weight C=50 in the objective wTw/2+C?ih(yixi?w).

% Trying PCA
CUTOFF = 100;
m = 1;
K=10
[coeff,score,latent] = pca(Xtrain);
Xtrain = score(:,1:CUTOFF);
c = cvpartition(275,'KFold',K);
opts = optimset('TolX',5e-4,'TolFun',5e-4);
Xtest = Xtest * coeff(:,1:CUTOFF);
select01 = (Ytrain==0)|(Ytrain==1);


minfn = @(z)kfoldLoss(fitcsvm(Xtrain(select01,:),Ytrain(select01),'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

% fval = zeros(m,1);
% z = zeros(m,2);
% for j = 1:m;
%     [searchmin fval(j)] = fminsearch(minfn,randi(200,2,1),opts);
%     z(j,:) = exp(searchmin);
% end
% z = z(fval == min(fval),:)
z = randi(200,2,1)
z(1)=100+randi(500,1,1)
z(2)=randi(30,1,1)
svm01 = fitcsvm(Xtrain(select01,:),Ytrain(select01), ...
    'KernelFunction', 'rbf','KernelScale','auto','BoxConstraint',z(2));
pred01 = predict(svm01, Xtest);

pred01;

c = cvpartition(378,'KFold',K);

select03 = (Ytrain==0)|(Ytrain==3);

minfn = @(z)kfoldLoss(fitcsvm(Xtrain(select03,:),Ytrain(select03),'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

% fval = zeros(m,1);
% z = zeros(m,2);
% for j = 1:m;
%     [searchmin fval(j)] = fminsearch(minfn,randi(200,2,1),opts);
%     z(j,:) = exp(searchmin);
% end
% z = z(fval == min(fval),:)
z = randi(200,2,1)
z(1)=100+randi(500,1,1)
z(2)=randi(30,1,1)
svm03 = fitcsvm(Xtrain(select03,:),Ytrain(select03), ...
    'KernelFunction', 'rbf','KernelScale','auto','BoxConstraint',z(2));
pred03 = predict(svm03, Xtest);
pred03;

c = cvpartition(349,'KFold',K);
select13 = (Ytrain==1)|(Ytrain==3);
minfn = @(z)kfoldLoss(fitcsvm(Xtrain(select13,:),Ytrain(select13),'CVPartition',c,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

% fval = zeros(m,1);
% z = zeros(m,2);
% for j = 1:m;
%     [searchmin fval(j)] = fminsearch(minfn,randi(200,2,1),opts);
%     z(j,:) = exp(searchmin);
% end
% z = z(fval == min(fval),:)
z = randi(200,2,1)
z(1)=100+randi(500,1,1)
z(2)=randi(30,1,1)
svm13 = fitcsvm(Xtrain(select13,:),Ytrain(select13), ...
    'KernelFunction', 'rbf','KernelScale','auto','BoxConstraint',z(2));
pred13 = predict(svm13, Xtest);
%% Test
% When testing, use majority vote among the three classifiers: e.g., if 
% the 0-1 classifier says 1, the 0-3 classifier says 3, and the 1-3 
% classifier says 1, predict 1.

% Try PCA
% Xtest = Xtest * coeff(:,1:CUTOFF);



pred13;
pred = zeros(nTest, nClass);
% pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
pred((pred01==0) & (pred03 ==0), 1) = 1;
pred((pred01==0) & (pred03 ==0), 2:3) = 0;
pred((pred01==1) & (pred13 ==1), 1) = 0;
pred((pred01==1) & (pred13 ==1), 2) = 1;
pred((pred01==1) & (pred13 ==1), 3) = 0;
pred((pred03==3) & (pred13 ==3), 1:2) = 0;
pred((pred03==3) & (pred13 ==3), 3) = 1;
pred_svm =pred;
% Write prediction.csv
save('pred.mat','pred_svm');
csvwrite('prediction.csv',pred);