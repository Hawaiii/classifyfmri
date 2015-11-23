load('../data/Train.mat');
% nnstart

CUTOFF = 200;
[coeff,score,latent] = pca(Xtrain);
trainCenter = mean(Xtrain,1);
Xtrain = score(:,1:CUTOFF);
Ytrain(Ytrain == 3) = 2;

inputs = Xtrain';
targets = Ytrain';

bestp = 0;
for i = 1:300
% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 5/100;
 
% Train the Network
[net,tr] = train(net,inputs,targets);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
 
% View the Network
% view(net)
if performance > bestp,
    load('../data/Test.mat');
    nTest = size(Xtest,1);
    nClass = 3;
    Xtest = (Xtest-repmat(trainCenter, nTest, 1))*coeff(:,1:CUTOFF);
    outputTest = net(Xtest');
    outputTest(outputTest < 0.5) = 0;
    outputTest(outputTest >= 0.5 & outputTest < 1.5) = 1;
    outputTest(outputTest >= 1.5) = 2;

    outputTest = outputTest';
    pred = zeros(nTest, nClass);
    pred(:,1) = 1; %hack, autolab complains about sum not equaling 1
    pred(outputTest==1, 1) = 0;
    pred(outputTest==1, 2) = 1;
    pred(outputTest==1, 3) = 0;
    pred(outputTest==2, 1:2) = 0;
    pred(outputTest==2, 3) = 1;

    % Write prediction.csv
    csvwrite('prediction.csv',pred);
    
    bestp = performance
end
end