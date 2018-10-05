%%  IDBP for inpainting: "Romeo and Juliet" superimposed text removal

% Reference: "Image Restoration by Iterative Denoising and Backward Projections"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Transactions on Image Processing (accepted 2018)

clear;
addpath('utilities');

denoiser_choice_index = 1;
denoiser_options = {'BM3D';'IRCNN'}; % These are two denoisers used in the TIP paper.
                                     % You can add your preferred denoiser, with the required support in the "Denoising_operation" file.
denoiser_choice = denoiser_options{denoiser_choice_index};
switch denoiser_choice
    case 'BM3D'
        addpath('Denoisers_folder\BM3D');
    case 'IRCNN'
        run('D:\Documents_D\MATLAB\matconvnet-1.0-beta25\matlab\vl_setupnn.m');
        addpath('Denoisers_folder\IRCNN');
        addpath('Denoisers_folder\IRCNN\utilities');
end


%% prepare observations

X0 = imread('test_sets\classics\4_Lena512.png');
scenario_ind = 2;
    
[M,N] = size(X0);
n = N*M;
X0 = double(X0);

switch scenario_ind
    case 1
        sig_e = 0;
        maxIter = 150; if strcmp(denoiser_choice,'IRCNN'); maxIter = 30; end;
        % for sig_e=0, delta=1-2 is the best, but then IDBP requires more iterations
        delta = 5; if strcmp(denoiser_choice,'IRCNN'); delta = 10; end;
    case 2
        sig_e = 10;
        maxIter = 75; if strcmp(denoiser_choice,'IRCNN'); maxIter = 30; end;
        delta = 0;
    case 3
        sig_e = 12;
        maxIter = 75; if strcmp(denoiser_choice,'IRCNN'); maxIter = 30; end;
        delta = 0;
end

rand('seed', 0);
randn('seed', 0);


[Y_clean,missing_pixels_ind] = put_text_for_inpainting(X0);

noise = sig_e * randn(M,N);
Y = Y_clean + noise;
Y(missing_pixels_ind) = 0;

Y_red = Y; Y_red(missing_pixels_ind)=255;
Y_red = cat(3,Y_red,Y,Y);
figure; imshow(uint8(Y_red));


%% run IDBP inpainting

% initialization by median scheme, feel free to check other options
X_median_init = median_inpainting(Y,missing_pixels_ind);
Y_tilde = X_median_init;
X_tilde = X_median_init;
sigma_alg = sig_e + delta;
if strcmp(denoiser_choice,'IRCNN')
    sigma_alg = floor(sigma_alg) + (mod(sigma_alg,2)<1); % IRCNN is trained only for odd sigma
end

for k=1:1:maxIter
    
    % estimate X_tilde
    %%% IMPORTANT:
    % note that "Denoising_operation" is written for treating the denoiser as a "black box"
    % and not for the fastest performance (e.g. it may load the same DNN and/or transfer data between CPU and GPU in each iteration)
    X_tilde = Denoising_operation(Y_tilde,sigma_alg,denoiser_choice);
    if max(X_tilde(:))<=1; X_tilde = X_tilde*255; end;
    
    % estimate Y_tilde
    Y_tilde = Y;
    Y_tilde(missing_pixels_ind) = X_tilde(missing_pixels_ind);
    
    % compute PSNR
    X_tilde_clip = X_tilde; X_tilde_clip(X_tilde<0) = 0; X_tilde_clip(X_tilde>255) = 255;
    PSNR = 10*log10(255^2/mean((X0(:)-X_tilde_clip(:)).^2));
    disp(['IDBP: finished iteration ' num2str(k) ', PSNR for X_tilde = ' num2str(PSNR)]);
    
end

if sig_e == 0 % i.e. if scenario_ind==1
    % in the noiseless case, take the last Y_tilde as the estimation
    X_tilde = Y_tilde;
    X_tilde_clip = X_tilde; X_tilde_clip(X_tilde<0) = 0; X_tilde_clip(X_tilde>255) = 255;
    PSNR = 10*log10(255^2/mean((X0(:)-X_tilde_clip(:)).^2));
    disp(['IDBP (noiseless case): finished iteration ' num2str(k) ', PSNR for X_tilde = ' num2str(PSNR)]);
end

figure; imshow(uint8(X_tilde));



