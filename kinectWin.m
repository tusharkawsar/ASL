%Hand Gesture Recognition For Sign Language.
%Using Microsoft Kinect.
%For Arithmetic operation.

clc;
clear all;
close all;

vid = videoinput('kinect', 1, 'RGB_640x480');
%vid = videoinput('kinect', 2, 'Depth_640x480');
src = getselectedsource(vid);

%Number of frame for passing
vid.FramesPerTrigger = inf;

%Video optimization for acquiring image from interest region
%ROI position [X Y W H]
vid.ROIPosition = [356 86 250 274];
preview(vid);

%Background frame
IM1=getsnapshot(vid);
pause(5);
for i=201:500
    pause(1);
    IM2=getsnapshot(vid);
    img = imsubtract(IM1,IM2);
    imgGray = rgb2gray(img);
    FilteredImage = medfilt2(imgGray, [3 3]);
    filename = strcat('five_',num2str(i),'.jpg')
    imwrite(FilteredImage,filename);
      
end

stoppreview(vid);
clc;
clear all;
close all;







