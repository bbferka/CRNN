function [combineAcc rgbAcc depthAcc] = runCRNN()
% init params
params = initParams();
disp(params);

rgbTrain_path = 'rgb_train.mat';
depthTrain_path = 'depth_train.mat';
%% Run RGB
disp('Forward propagating RGB data');
parmas.depth = false;

if(exist(rgbTrain_path,'file'))
    % load and forward propagate RGB data
    [rgbTrain rgbTest] = forwardProp(params, true);
    load(rgbTrain_path);
else
    [rgbTrain rgbTest] = forwardProp(params, false);
    save(rgbTrain_path, 'rgbTrain', '-v7.3');
end

% train softmax classifier
disp('Training softmax...');
rgbAcc = trainSoftmax(rgbTrain, rgbTest, params,'rgb');

%% Run Depth
disp('Forward propagating depth data');
params.depth = true;

if(exist(depthTrain_path,'file'))
    % load and forward propagate depth data
    [depthTrain depthTest]  = forwardProp(params, true);
    load(depthTrain_path);
else
    [depthTrain depthTest] = forwardProp(params, false);
    save(depthTrain_path, 'depthTrain', '-v7.3');
end

% train softmax classifier
depthAcc = trainSoftmax(depthTrain, depthTest, params,'depth');

%% Combine RGB + Depth
[cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest);
clear rgbTrain rgbTest depthTrain depthTest;

% test without extra features when combined
params.extraFeatures = false;
save('finalFeatures.mat','cTrain','cTest', '-v7.3');
combineAcc = trainSoftmax(cTrain, cTest, params,'both');
return;

function [train test] = forwardProp(params, skipTraining)
% pretrain filters
disp('Pretraining CNN Filters...');
[filters params] = pretrain(params);

% forward prop CNN
disp('Forward prop through CNN...');
[train test] = forwardCNN(filters,params, skipTraining);

% forward prop RNNs
disp('Forward prop through RNN...');
[train test] = forwardRNN(train, test, params, skipTraining);
return;

function [cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest)
% ensure they come from the same file
testCompatability(rgbTrain, depthTrain);
testCompatability(rgbTest, depthTest);

% combine data
cTrain.data = [rgbTrain.data; depthTrain.data];
cTest.data = [rgbTest.data; depthTest.data];

% normalize depth and rgb features independently
m = mean(cTrain.data,2);
s = std(cTrain.data,[],2);
cTrain.data = bsxfun(@rdivide, bsxfun(@minus, cTrain.data,m),s);
cTest.data = bsxfun(@rdivide, bsxfun(@minus, cTest.data,m),s);

% add the labels
cTrain.labels = rgbTrain.labels;
cTest.labels = rgbTest.labels;
return;

function testCompatability(rgb, depth)
assert(length(rgb.file) == length(depth.file));
for i = 1:length(rgb.file)
    assert(strcmp(rgb.file{i}, depth.file{i}));
end

assert(isequal(rgb.labels, depth.labels));
assert(isequal(rgb.labels, depth.labels));
return
