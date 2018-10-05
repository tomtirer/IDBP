%%  IDBP for super-resolution

% Reference: "Image Restoration by Iterative Denoising and Backward Projections"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Transactions on Image Processing (accepted 2018)

% NOTE: 
% Tuned for Gaussian 7x7 kernel, standard deviation 1.6, and scale factor 3, for two noise levels: {0, sqrt(2)}.
% Automatic tuning is not supported yet, but should be similar to the deblurring case.

clear;
addpath('utilities');

denoiser_choice_index = 1;
dataset_choice_index = 1;

denoiser_options = {'BM3D';'IRCNN'}; % These are two denoisers used in the TIP paper.
                                     % You can add your preferred denoiser, with the required support in the "Denoising_operation" file.
dataset_options = {'classics','BSD68','Set5','Set14'};


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
images_folder = ['test_sets\' dataset_choice];
ext                 =  {'*.jpg','*.png','*.bmp'};
images_list           =  [];
for i = 1 : length(ext)
    images_list = cat(1,images_list,dir(fullfile(images_folder, ext{i})));
end
N_images = size(images_list,1);


all_results_PSNR = zeros(2,N_images);
all_results_ssim = zeros(2,N_images);

for scenario_ind=1:1:2
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
        X0 = double(X0);
        
        s = 3; % scale        
        h = fspecial('gaussian', 7, 1.6);
        Hfunc = @(Z) downsample2(imfilter(Z,h,'conv','replicate'),s);
        Htfunc = @(Z) imfilter(upsample2_MN(Z,s,[M,N]),fliplr(flipud(conj(h))),'conv','replicate');
        
        switch scenario_ind
            case 1
                % scenario 1
                sig_e = 0;
            case 2
                % scenario 2
                sig_e = sqrt(2);
        end
        
        
        Y_clean = Hfunc(X0);
        [Mlr,Nlr] = size(Y_clean);
        
        randn('seed',0);
        noise = sig_e * randn(Mlr,Nlr);
        Y = Y_clean + noise;
        
        Y_upscaled = imresize(Y,[M,N],'bicubic');
        input_PSNR = 10*log10(255^2/mean((X0(:)-Y_upscaled(:)).^2))

        
        %% run IDBP super-resolution
        
        maxIter = 30;
        use_different_delta = 1; delta = 13; % for sig_e=sqrt(2), fixed delta=13 (LHS/RHS ~2) can be used with minor performance loss
        use_different_epsilon = 0; epsilon = 0;
        
        if sig_e==0; delta_add = 0; else delta_add = 11-s; end;
        delta_list = delta_add + logspace(log10(12*s),log10(s),maxIter);
        epsilon_list = logspace(log10(70e-3),log10(7e-3),maxIter);
        
        % initialization
        Y_tilde = Y_upscaled; X_tilde = Y_upscaled;
        sigma_alg = sig_e + delta;
        if strcmp(denoiser_choice,'IRCNN')
            sigma_alg = floor(sigma_alg) + (mod(sigma_alg,2)<1); % IRCNN is trained only for odd sigma
            delta = sigma_alg - sig_e;
        end
        epsilon_sig2_e = epsilon*sig_e^2;
        HHt_cg = @(z) vec(Hfunc(Htfunc(reshape(z,Mlr,Nlr))))+z*epsilon_sig2_e;
        
        for k=1:1:maxIter
            
            if use_different_delta
                sigma_alg = sig_e + delta_list(k);
            end
            if use_different_epsilon
                epsilon_sig2_e = epsilon_list(k)*sig_e^2;
                HHt_cg = @(z) vec(Hfunc(Htfunc(reshape(z,Mlr,Nlr))))+z*epsilon_sig2_e;
            end
                        
            % estimate Y_tilde
            H_X_tilde = Hfunc(X_tilde);
            [cg_result, iter, residual] = cg(zeros(Mlr*Nlr,1), HHt_cg, vec(Y-H_X_tilde), 100, 10^-6); % cg_result = inv(H*Ht)*(Y-H*X_tilde)
            if sqrt(residual)>10^-3
                disp(['cg: finished after ' num2str(iter) ' iterations with norm(residual) = ' num2str(sqrt(residual)) ' - Use preconditioning or tikho regularization (epsilon) for HHt_cg']);
            end
            Y_tilde = Htfunc(reshape(cg_result,Mlr,Nlr)) + X_tilde;
            
            % estimate X_tilde
            % note that "Denoising_operation" is written for treating the denoiser as a "black box"
            % and not for the fastest performance (e.g. it may load the same DNN and copy between CPU and GPU in each iteration)
            X_tilde = Denoising_operation(Y_tilde,sigma_alg,denoiser_choice);
            if max(X_tilde(:))<=1; X_tilde = X_tilde*255; end;            
            
            % checking Proposition 1
            temp1 = Y-Hfunc(X_tilde);
            [temp2b, iter, residual] = cg(zeros(Mlr*Nlr,1), HHt_cg, temp1(:), 100, 10^-6);
            temp2 = Htfunc(reshape(temp2b,Mlr,Nlr));
            Prop1_LHS = temp1(:)'*temp1(:)/sig_e^2;
            Prop1_RHS = temp2(:)'*temp2(:)/(sigma_alg)^2;
            
            % compute PSNR
            X_tilde_clip = X_tilde; X_tilde_clip(X_tilde<0) = 0; X_tilde_clip(X_tilde>255) = 255;
            PSNR = 10*log10(255^2/mean((X0(:)-X_tilde_clip(:)).^2));
            disp(['IDBP: finished iteration ' num2str(k) ', PSNR for X_est = ' num2str(PSNR) ', LHS/RHS ratio (ignore in noiseless case) = ' num2str(Prop1_LHS/Prop1_RHS)]);
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


