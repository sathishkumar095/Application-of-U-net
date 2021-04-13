clear all;
close all;
clc;
[file path]=uigetfile('.dcm','choose any image');
a=dicomread([path file]);
figure(1);
imshow(a,[]);
title('Input Image');

% close all; clear all
% clc
% 
% srcFile = dir('E:\dataset\Datasets\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566-03192\*.dcm');

% for i=1:length(path file)
%     filename= strcat('E:\dataset\Datasets\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566-03192\',srcFile(i).name);
    
% end


b=imcrop
figure(2),imshow(b)
global b
imageSize = [512 512 1];
numClasses = 2;
encoderDepth = 2;

segmat