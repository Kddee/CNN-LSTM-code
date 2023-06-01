
clc;
clear;
x = readtable('Updated DATA for ALE Phata 2.xlsx');
load('trained_load.mat')
Feature=table2array(x(:,3:9)); %100 X
Response=table2array(x(:,10)); %Y
numFeatures =7;
numResponses = 1;
mu1 = mean(Response); 
sig1 = std(Response); 
dataTrainStandardized = (Response - mu1) / sig1; 
%%

figure
plot(Response)
xlabel("time ")
ylabel("Load )")
title(" july 2020 to august 2022 Load")
cv = cvpartition(size(Feature,1),'HoldOut',0.3);
idx = cv.test;
%data  standarizing and generalization
%features train
XTrain = Feature(~idx,:);
%features test
XTest  = Feature(idx,:);
%mean of features train
muxt = mean(XTrain); 
%standard deviation of features train
sigxt = std(XTrain); 
%normalizing data between -1 to 1
XTrain = ((XTrain - muxt) ./ sigxt)'; 
%%
%
%mean of the features test
muxe = mean(XTest); 
%mean of the features test
sigxe = std(XTest); 
%normalizing data between -1 to 1
XTest = ((XTest - muxe) ./ sigxe)'; 
%
%dividing responses data into training and testing
cv = cvpartition(size(Response,1),'HoldOut',0.3);
idx = cv.test;
%responses training
YTrain = Response(~idx,:);
%responses testing
YTest  = Response(idx,:);
%mean train responses
mu1 = mean(YTrain); 
%mean training responses
sig1 = std(YTrain); 
%generalizing training responses between -1 to 1
YTrain = ((YTrain - mu1) ./ sig1)'; 
%
%mean testing responses
muye = mean(YTest); 
%standard testing responses
sigye = std(YTest); 
%normalizing testing responses between -1 to 1
YTest = ((YTest - muye) ./ sigye)'; 
%saving testing responses in mat file
save('YTest.mat','YTest')
numHiddenUnits = 100; 
%%loa
% layers = [
%     sequenceInputLayer(numFeatures,"Name","input")
%     convolution1dLayer(11,96,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
% %     convolution1dLayer(20,180,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     
% %     convolution1dLayer(30,300,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     convolution1dLayer(32,320,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     dropoutLayer(0.2)
%    bilstmLayer(100,'OutputMode','sequence') 
% %     dropoutLayer(0.1) 
% %    bilstmLayer(105,'OutputMode','sequence') 
% %     dropoutLayer(0.2) 
% %     bilstmLayer(110,'OutputMode','sequence') 
% %     dropoutLayer(0.2) 
%     fullyConnectedLayer(1)
%     regressionLayer];
% options = trainingOptions("adam",...
%     "GradientThreshold",1,...
%     "InitialLearnRate",0.001,...
%     'MaxEpochs',50,...
%     'SequenceLength','longest',...
%     'Epsilon',1e-8,...
%     'L2Regularization',0.0001,...
%     "Shuffle","every-epoch",...
%     'GradientDecayFactor',0.9,...
%     'SquaredGradientDecayFactor',0.999,...
%     'LearnRateDropFactor',0.1,...
%     'LearnRateDropPeriod',10,...
%     'GradientThresholdMethod','l2norm',...
%     'ResetInputNormalization',true,...
%     "Plots","training-progress");
% net = trainNetwork(XTrain,YTrain,layers,options);
% save('trained_load.mat','net')
% save('Test.mat','XTest')
%predicting responses from model
YPred=predict(net,XTest)
%error between actual and predicted values
rmse = sqrt(mean((YPred' - YTest').^2)); 
%converting predicting data back to original form
YPred = sigye*YPred + muye;
%converting actual data back to original form
YTest = sig1*YTest + mu1;
%%
%saving data into mat file
save('mean.mat','mu1')
save('standard.mat','sig1')
YPred=double(YPred);

csvwrite('predicted_load.csv',YPred)
%plotting actual and predicted plots
figure
plot(YTest')
hold on
plot(YPred')
xlabel("Time ")
ylabel("Load")
title("short term Last quarter forecasting")
legend(["Observed" "Forecast"])
hold off
%

