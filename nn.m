%%
% Neural Network for classification of microarray data
% BMES 543 / 483
% Authors: Yang Wan, Anna Lu
% Last modified: June 5, 2017


%% Load Data
data = xlsread('nci.xlsx');

%% Randomize the data
rng(0);
datacols = size(data, 2);
randX = randperm(datacols); % randomize data, columnwise

% 2/3 cutoff for training and 1/3 remaining for testing
data_cutoff = ceil(2/3 * datacols);
train_ind = randX(1:data_cutoff);
test_ind = randX(data_cutoff+1:end);

clear data_cutoff;

% select training input and targets
inputs = data(train_ind);
targets = data(1, train_ind);

%% network training
net=patternnet(10);
net.trainParam.showWindow=false;
%net.divideParam.trainRatio = 0.8;
%net.divideParam.valRatio = 0.2;
%net.divideParam.testRatio = 0;
%[net] = train(net,transpose(class1),transpose(class2));
[net, tr] = train(net,inputs,targets);

%% network testing

output = net(data(test_ind));

%% network evaluation

% error rate
actual = data(1, test_ind);
e = actual - output;
mean_err = mean(e); % n = 12
ploterrhist(e, 'bins', 5)

%% performance
perform(net, actual, output) % error weight default {1}
plotperf(tr)

%% visualize
view(net)