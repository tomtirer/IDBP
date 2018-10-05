%%  IDBP for inpainting

% Reference: "Image Restoration by Iterative Denoising and Backward Projections"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Transactions on Image Processing (accepted 2018)

clear;
addpath('utilities');

denoiser_choice_index = 1;
dataset_choice_index = 1;

denoiser_options = {'BM3D';'IRCNN'}; % These are two denoisers used in the TIP paper.
                                     % You can add your preferred denoiser, with the required support in the "Denoising_operation" file.
dataset_options = {'classics','BSD68'};


denoiser_choice = denoiser_options{denoiser_choice_index};
switch denoiser_choice
    case 'BM3D'
        addpath('Denoisers_folder\BM3D');
    case 'IRCNN'
        run('D:\Documents_D\MATLAB\matconvnet-1.0-beta25\matlab\vl_setupnn.m');
        addpath('Denoisers_folder\IRCNN');
        addpath('Denoisers_folder\IRCNN\utilities');
end

dataset_choice = dataset_options{dataset_choice_index};
dataset_choice = dataset_options{dataset_choice_index};
images_folder = ['test_sets\' dataset_choice];
ext                 =  {'*.jpg','*.png','*.bmp'};
images_list           =  [];
for i = 1 : length(ext)
    images_list = cat(1,images_list,dir(fullfile(images_folder, ext{i})));
end
N_images = size(images_list,1);


all_results_PSNR = zeros(3,N_images);
all_results_ssim = zeros(3,N_images);

for scenario_ind=1:1:3
    for image_ind=1:1:N_images
        %% prepare observations
        
        img_name = images_list(image_ind).name;
        X0 = imread(fullfile(images_folder,img_name));
        
        assert(max(X0(:))<=255);
        if size(X0,3)>1
            X0 = rgb2ycbcr(X0);
            X0 = X0(:,:,1);
        end
        
        if strcmp(img_name,'3_peppers256.png')
            X0 = X0(2:end-1,2:end-1); % remove defective intensity values
        end
        
        [M,N] = size(X0);
        n = N*M;
        p = floor(0.8*n); if strcmp(dataset_options,'BSD68'); p = floor(0.5*n); end;
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
        
        missing_pixels_ind = randperm(n,p);
        Y_clean = X0;
        Y_clean(missing_pixels_ind) = 0;
        
        noise = sig_e * randn(M,N);
        Y = Y_clean + noise;
        Y(missing_pixels_ind) = 0;
        
        
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
            % note that "Denoising_operation" is written for treating the denoiser as a "black box"
            % and not for the fastest performance (e.g. it may load the same DNN and copy between CPU and GPU in each iteration)
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
        
        
        %% collect results
        
        all_results_PSNR(scenario_ind,image_ind) = PSNR;
        ssim_res = ssim(double(X_tilde_clip)/255,double(X0)/255); % we use MATLAB R2016a function
        all_results_ssim(scenario_ind,image_ind) = ssim_res;
        disp(['scenario_ind=' num2str(scenario_ind) ', image_ind=' num2str(image_ind) ', PSNR=' num2str(PSNR) ', SSIM=' num2str(ssim_res)]);
        
    end
end



