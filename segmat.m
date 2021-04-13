global b imds options layers
inp=b;


DatasetPath=fullfile('E:\452\src\dataset1');
imds=imageDatastore(DatasetPath, 'IncludeSubfolders', true,...
    'LabelSource','foldernames','fileextension',{'.dcm'});

labelDir = fullfile(DatasetPath,'testImages');


I = readimage(imds,1);
I = histeq(I);
imshow(I)

classes = [
    "MALIGNANT","BENIGN"
    ];

labelIDs=[255 0];

inputlayer = imageInputLayer([512 512 1],'Name','inp');
numFilters = 64;
numLayers = 16;
layers = [ ...
    imageInputLayer([512 512 1])
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
     convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(5,20) 
    fullyConnectedLayer(2)
    softmaxLayer
    pixelClassificationLayer
    ];

% %%  denoising image data set
% dnimds=denoisingImageDatastore(imds,'patchSize', [224 224], 'PatchesPerImage', 64, 'ChannelFormat','grayscale');
%% 
% pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',false);
%% 

net=trainNetwork(imds,layers,options);

% res = activations(net,inp,net.Layers(numLayers-1).Name,'OutputAs','channels');

 % Read image and pixel label data. read(pxds) returns a categorical
  % matrix, C. C(i,j) is the categorical label assigned to I(i,j).
  I = read(imds);
%   C = read(pxds);
%   
  % Display the label categories in C
%   categories(I)
  C = semanticseg(I, net);
  % Overlay pixel label data on the image and display.
  B = labeloverlay(I, C);
%   t = graythresh(b);
%   bw = imbinarize(b,t);
%   imshow(bw)
  figure(12)
  imshow(B)
  