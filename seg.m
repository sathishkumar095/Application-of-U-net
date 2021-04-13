
A1 = 'C:\Users\sathi\Desktop\image\label';
dataSetDir = fullfile(A1,'dataset');

% dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
% imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'Label');
imds= imageDatastore(Datasetpath,'IncludeSubfolders', true , 'LabelSource','foldernames','FileExtension',{'.dcm'});
% Data=imageDatastore


%  imds = imageDatastore(imageDir);
classNames = ["benign","malignant"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
imageSize = [512 512 1];
 numClasses = 2;
lgraph = createUnet(imageSize, numClasses);
ds = pixelLabelImageDatastore(imds,pxds);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'VerboseFrequency',10);
net = trainNetwork(imds,lgraph,options);