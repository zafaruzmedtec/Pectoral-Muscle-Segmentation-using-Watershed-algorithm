    % Pectoral Muscle Segmentation

% IMPORTANT: Before running the code, please follow these steps:
    % - the code reads 'dataset' folder;
    % - 'images', 'groundthruths' and 'masks' folders should be inside the
    % 'dataset' folder;
    % - to save the segmented images, create a folder named "segmented"
    % inside the 'dataset' folder;

clear
close all
clc

% Loads original images, groundtruths and masks, having the given file extension
dataset = 'dataset';
images = dir(fullfile(dataset,'/images/','*.tif'));
groundtruths = dir(fullfile(dataset,'/groundtruths/','*.tif'));
masks = dir(fullfile(dataset,'/masks/','*.png'));
figure;

% loop to read all images one by one
for i = 1:length(images)
    original_img = imread(fullfile(dataset,'/images/',images(i).name));
    groundtruths_img = imread(fullfile(dataset,'/groundtruths/',groundtruths(i).name));
    masks_img = imread(fullfile(dataset,'/masks/',masks(i).name));
    %figure; imshow(original_img); title('Original image')
    
    % Histogram equalization
    hist = histeq(original_img);
    % figure; imshow(hist); title('Histogram equalization')

    % Gaussian smoothing (sigma = 12)
    gauss_smooth  = imgaussfilt(hist, 12); 
    % figure; imshow(gauss_smooth); title('Gaussian smoothing')
    
    % Edge detection using gradient magnitude
    hy = fspecial('sobel'); % using sobel filter
    hx = hy';
    dy = imfilter(double(gauss_smooth), hy, 'replicate');
    dx = imfilter(double(gauss_smooth), hx, 'replicate');
    grad_mag = sqrt(dx.^2 + dy.^2);
    % figure(); imshow(grad_mag,[]), title('Gradient magnitude')
    
    % Creates a black image for markers
    [rows, columns] = size(original_img); % gets rows and columns of original image
    markers = zeros(rows, columns); % create black image using rows and columns of original image

    % Creates two markers on the black image
    markers(1:50,1:50) = 65536; % marker 1 (internal) in the top left corner pectoral muscle region (65536 means white color in 16 bit image)
    %The size of the input images are different and the size of second 
    %marker (in breast profile) are correspondingly smaller than the image
    markers(rows/2.5:(rows/2.5)+1000, columns/4:(columns/4)+1000) = 65536; % marker 2 (external) in the breast profile
    
    % If breast is on the left, do nothing otherwise flip the markers 
    if original_img(30, 30) ~= 0 % if 30x30 pixel intensity is not black
        markers = markers;
    else
        markers = fliplr(markers); % flips the markers
    end
    %figure; imshow(markers); title('Markers image')

    % Obtains minima for watershed transformation
    impose_minima = imimposemin(grad_mag, markers); % minima imposed in two location markers on gradient magnitude image
    %figure; imshow(impose_minima,[]); title('Minima imposed')
    
    % Watershed transformation
    water_sh = watershed(impose_minima);
    
    % If 30x30 pixel is not black, replaces all pixels to 1 (label 1)
    % otherwise to 2 (label 2)
    if original_img(30, 30)~=0
        pect_muscle = (water_sh == 1); %label 1
    else pect_muscle = (water_sh == 2); %label 2
    end
    %figure; imshow(pect_muscle); title('Pectoral muscle')
    
    pect_muscle = logical(pect_muscle); %converts to logical values (binary image)
    
    % Plots GroundTruth and Pectoral Muscle Segmentation
    subplot 121
    imshow(groundtruths_img); title('GroundTruth')
    subplot 122
    imshow(pect_muscle);
    str = sprintf('Pectoral Muscle Segmentation, Img: %d ', i); title(str);
    pause(0.1);
    
    % Converts the segmented image to 8 bit
    pect_muscle = im2uint8(pect_muscle);
    
    % Path into which the segmented image will be saved with the same original name
    fullFileName = fullfile(dataset,'/segmented/',images(i).name);
    
    % Saves the segmented image
    imwrite(pect_muscle, fullFileName);
end


     %% Compute Segmentation Accuracy
     
% Loads segmented images, groundtruths and masks, having the given file extension
dataset = 'dataset';
segmented = dir(fullfile(dataset,'/segmented/','*.tif'));
groundtruths = dir(fullfile(dataset,'/groundtruths/','*.tif'));
masks = dir(fullfile(dataset,'/masks/','*.png'));
Accuracy = 0; %starts from zero

% Reads all images one by one
for j = 1:length(segmented)
    segmented_img = imread(fullfile(dataset,'/segmented/',segmented(j).name));
    groundtruths_img = imread(fullfile(dataset,'/groundtruths/',groundtruths(j).name));
    masks_img = imread(fullfile(dataset,'/masks/',masks(j).name));
    segmented_img = im2uint8(segmented_img); % converts segmented image to 8 bit
    ACC = 1 - sum(sum(abs(groundtruths_img - segmented_img)))/sum(sum(masks_img)); %calculates ACC = (TP + TN)/total_number_of_samples
    Accuracy = Accuracy + ACC; %adds each ACC
end

Accuracy = Accuracy/length(segmented); %gets ACC for all images
print_acc = [' Accuracy: ', num2str(Accuracy)]; %prints
disp(print_acc);