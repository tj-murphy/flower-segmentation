%%
clear
close all;

%% Prepare data
imageDir = '/images_256';
labelDir = '/labels_256';

% Get list of image and label files using dir function
imageFiles = dir(fullfile(imageDir, '*.jpg'));
labelFiles = dir(fullfile(labelDir, '*.png'));

% Extract image and label file names
imageNames = {imageFiles.name};
labelNames = {labelFiles.name};

% Find matching image and label pairs
% I used intersect function to find the common elements between imageNames
% and label Names
[~, imageIndices, labelIndices] = intersect(imageNames, strrep(labelNames, '.png', '.jpg'));

% Create image and label file paths
% Combine dir paths with corresponding filenames using fullfile
imagePaths = fullfile(imageDir, imageNames(imageIndices));
labelPaths = fullfile(labelDir, labelNames(labelIndices));

%% Split data
% I took the following approach because I initially had trouble splitting
% the data into train, test, and val sets. It was difficult to split once
% the datastores had been created, so this approach splits it first, then
% creates datastores.

trainRatio = 0.8;  % I wanted to maximise training data due to size of the dataset
valRatio = 0.1;
testRatio = 0.1;

% Find total no. of images and shuffle using random permutation
numImages = numel(imagePaths);
shuffledIndices = randperm(numImages);  % randperm = no repeated elements

% Split shuffled indices into training, val and test indices using ratio
trainIndices = shuffledIndices(1:round(trainRatio * numImages));
valIndices = shuffledIndices(round(trainRatio * numImages)+1:round((trainRatio + valRatio) * numImages));
testIndices = shuffledIndices(round((trainRatio + valRatio) * numImages)+1:end);

% Create datastores for training, val and test sets
% Image datastores contain file paths to images
% Pixel label datastores contain file paths to label images, class names,
% label indices.
trainImds = imageDatastore(imagePaths(trainIndices));
trainPxds = pixelLabelDatastore(labelPaths(trainIndices), ["flower", "background"], [1,3]);

valImds = imageDatastore(imagePaths(valIndices));
valPxds = pixelLabelDatastore(labelPaths(valIndices), ["flower", "background"], [1,3]);

testImds = imageDatastore(imagePaths(testIndices));
testPxds = pixelLabelDatastore(labelPaths(testIndices), ["flower", "background"], [1,3]);

% Combine image and label datastores
trainDS = combine(trainImds, trainPxds);
valDS = combine(valImds, valPxds);
testDS = combine(testImds, testPxds);

%% Model architecture
inputSize = [256 256 3];  % Size of images
numClasses = 2;

layers = [
	% Encoder path
    % Downsamples feature maps.
    imageInputLayer(inputSize)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    dropoutLayer(0.5)

	% Decoder path
    % Upsamples feature maps and recovers spatial resolution.
    transposedConv2dLayer(4, 512, 'Stride', 2, 'Cropping', 'same')
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    
    transposedConv2dLayer(4, 256, 'Stride', 2, 'Cropping', 'same')
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    
    transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same')
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    
    % Final layers
    convolution2dLayer(1, numClasses, 'Padding', 'same')
    softmaxLayer()
    pixelClassificationLayer()
];

%% Train
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 5, ...  
    'MiniBatchSize', 64, ...                    % Optimal trade off between memory and efficiency
    'Shuffle', 'every-epoch', ...            % Shuffle training data at every epoch to reduce overfitting and improve generalisability
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', valDS, ...            % Use validation dataset to evaluate during training
    'ValidationFrequency', 5, ...
    'ValidationPatience', 3);               % If performance doesn't improve after 3 iterations, stop training early

net = trainNetwork(trainDS, layers, options);

%% Save model
save('segmentownnet.mat', 'net');

%% Load model, if needed
load("segmentownnet.mat");

%% Evaluate model
pxdsResults = semanticseg(testDS, net, 'WriteLocation', tempdir, 'Verbose', false);

%% Calculate accuracy
metrics = evaluateSemanticSegmentation(pxdsResults, testDS, 'Verbose', false);

% Overall metrics
accuracy = metrics.DataSetMetrics.MeanAccuracy;
meanIoU = metrics.DataSetMetrics.MeanIoU;

% Per class metrics
classAccuracy = metrics.ClassMetrics.Accuracy;
classIoU = metrics.ClassMetrics.IoU;
classMeanBFScore = metrics.ClassMetrics.MeanBFScore;

% Display overall metrics
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Mean IoU: ', num2str(meanIoU)]);

% Display per class metrics
classNames = {"Flower", "Background"};
for i = 1:numel(classNames)
    disp([classNames{i}, ":"]);
    disp([" Accuracy:", num2str(classAccuracy(i))]);
    disp([" IoU:", num2str(classIoU(i))]);
    disp([" Mean BF Score:", num2str(classMeanBFScore(i))]);
end

%% Visualise
numTestImages = 4;

for i = 1:numTestImages
    % Get image and predicted mask
    testData = read(testDS);
    testImage = testData{1};
    testLabel = testData{2};
    predictedMask = readimage(pxdsResults, i);

    % Convert testLabel and predicted mask to numeric arrays
    testLabel = uint8(testLabel);
    predictedMask = uint8(predictedMask);

    % Display img and predicted mask
    figure;
    subplot(1,3,1);
    imshow(testImage);
    title('Test Image');

    subplot(1,3,2);
    imshow(testLabel, []);
    title('Ground Truth Label');

    subplot(1,3,3);
    imshow(predictedMask, []);
    title('Predicted Mask');
end

%% Confusion Matrix
confMatrix = table2array(metrics.ConfusionMatrix);
classLabels = {'flower', 'background'};
confChart = confusionchart(confMatrix, classLabels);
confChart.Title = 'Confusion Matrix - Custom';

confChart.RowSummary = 'row-normalized';
confChart.ColumnSummary = 'column-normalized';

%% Histogram of IoU values
IoUValues = metrics.ImageMetrics.MeanIoU;
figure;
histogram(IoUValues);
xlabel('IoU Value');
ylabel('Frequency');
title('segmentationOwn IoU Histogram');
