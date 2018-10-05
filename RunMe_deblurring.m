%%  IDBP for (non-blind) deblurring

% Reference: "Image Restoration by Iterative Denoising and Backward Projections"
% Authors: Tom Tirer and Raja Giryes
% Journal: IEEE Transactions on Image Processing (accepted 2018)

clear;
addpath('utilities');

flag_autoTune_IDBP = 0; % automatically tunes epsilon
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
images_folder = ['test_sets\' dataset_choice];
ext                 =  {'*.jpg','*.png','*.bmp'};
images_list           =  [];
for i = 1 : length(ext)
    images_list = cat(1,images_list,dir(fullfile(images_folder, ext{i})));
end
N_images = size(images_list,1);


all_results_ISNR = zeros(4,N_images);
all_results_ssim = zeros(4,N_images);

for scenario_ind=1:1:4
    for image_ind=1:1:N_images
        %% prepare observations
        
        img_name = images_list(image_ind).name;
        X0 = imread(fullfile(images_folder,img_name));
        
        assert(max(X0(:))<=255);
        if size(X0,3)>1
            X0 = rgb2ycbcr(X0);
            X0 = X0(:,:,1);
        end
        
        [M,N] = size(X0);
        X0 = double(X0);
        clear H;
        
        switch scenario_ind
            case 1
                % scenario 1
                s1=0; for a1=-7:7; s1=s1+1; s2=0; for a2=-7:7; s2=s2+1; H(s1,s2)=1/(a1^2+a2^2+1); end, end;  H=H./sum(H(:));
                sig_e = sqrt(2);
                epsilon = 7e-3; if strcmp(denoiser_choice,'IRCNN'); epsilon = 4e-3; end;
            case 2
                % scenario 2
                s1=0; for a1=-7:7; s1=s1+1; s2=0; for a2=-7:7; s2=s2+1; H(s1,s2)=1/(a1^2+a2^2+1); end, end;  H=H./sum(H(:));
                sig_e = sqrt(8);
                epsilon = 4e-3; if strcmp(denoiser_choice,'IRCNN'); epsilon = 2e-3; end;
            case 3
                % scenario 3
                H = ones(9,9); H = H./sum(H(:));
                sig_e = sqrt(0.308);
                epsilon = 8e-3; if strcmp(denoiser_choice,'IRCNN'); epsilon = 3e-3; end;                
            case 4
                % scenario 4
                H = [1 4 6 4 1]'*[1 4 6 4 1]; H = H./sum(H(:));
                sig_e = sqrt(49);
                epsilon = 2e-3; if strcmp(denoiser_choice,'IRCNN'); epsilon = 0.8e-3; end;
        end
        
        
        Y_clean = cconv2_by_fft2(X0,H,0);
        
        if scenario_ind==3
            % in scenrio 3 we make sure that BSNR=40 (like in IDD-BM3D experiments)
            sig_e = sqrt(norm(Y_clean(:)-mean(Y_clean(:)),2)^2 /(N*N*10^(40/10)));
        end
        
        randn('seed',0);
        noise = sig_e * randn(M,N);
        Y = Y_clean + noise;
        
        BSNR = 10*log10(norm(Y_clean(:)-mean(Y_clean(:)),2)^2 /(N*N*sig_e^2))
        input_PSNR = 10*log10(255^2/mean((X0(:)-Y(:)).^2))
        
              
        %% run IDBP deblurring
                
        maxIter = 30; % 30 iterations used in the paper, often 15-20 are enough.
        delta = 5; if strcmp(denoiser_choice,'IRCNN'); delta = 10; end;
        sigma_alg = sig_e + delta;
        if strcmp(denoiser_choice,'IRCNN')
            sigma_alg = floor(sigma_alg) + (mod(sigma_alg,2)<1); % IRCNN is trained only for odd sigma
            delta = sigma_alg - sig_e;
        end

                
        if ~flag_autoTune_IDBP
            
            % initialization
            Y_tilde = Y; X_tilde = Y;
            epsilon_sig2_e = epsilon*sig_e^2; % set epsilon according to scenario
            
            for k=1:1:maxIter
                
                % estimate X_tilde
                % note that "Denoising_operation" is written for treating the denoiser as a "black box"
                % and not for the fastest performance (e.g. it may load the same DNN and copy between CPU and GPU in each iteration)
                X_tilde = Denoising_operation(Y_tilde,sigma_alg,denoiser_choice);
                if max(X_tilde(:))<=1; X_tilde = X_tilde*255; end;
                
                % estimate Y_tilde
                H_conv_X_tilde = cconv2_by_fft2(X_tilde,H,0,[]);
                Y_tilde = cconv2_by_fft2(Y-H_conv_X_tilde,H,1,epsilon_sig2_e) + X_tilde; 
                              
                % checking Proposition 1
                temp1 = Y-cconv2_by_fft2(X_tilde,H,0,[]);
                temp2 = cconv2_by_fft2(temp1,H,1,epsilon_sig2_e); % apply approximated H^dagger 
                Prop1_LHS = temp1(:)'*temp1(:)/sig_e^2;
                Prop1_RHS = temp2(:)'*temp2(:)/(sigma_alg)^2;
                                
                % compute PSNR and ISNR
                Y_clip = Y; Y_clip(Y<0) = 0; Y_clip(Y>255) = 255;
                X_tilde_clip = X_tilde; X_tilde_clip(X_tilde<0) = 0; X_tilde_clip(X_tilde>255) = 255;
                ISNR = 20*log10( norm(X0 - Y_clip, 'fro') / norm(X0 - X_tilde_clip, 'fro'));
                PSNR = 10*log10(255^2/mean((X0(:)-X_tilde_clip(:)).^2));
                disp(['IDBP: finished iteration ' num2str(k) ', PSNR for X_tilde = ' num2str(PSNR) ', ISNR = ' num2str(ISNR) ', LHS/RHS ratio = ' num2str(Prop1_LHS/Prop1_RHS)]);
            end
            
            
        else % flag_autoTune_IDBP == 1
            
            ratio_thr = 3; if strcmp(denoiser_choice,'IRCNN'); ratio_thr = 4; end;
            epsilon_inc = 1e-4;
            delta_inc = 0;
            epsilon = 0.5e-3;           
            epsilon_sig2_e = epsilon*sig_e^2;
            epsilon_sig2_e = max([epsilon_sig2_e, 5e-4]); % protection against sig_e near 0
            sigma_alg = sig_e + delta;
            flag_IDBP_finished = 0;
            
            while ~flag_IDBP_finished
                
                % initialization per new epsilon
                Y_tilde = Y; X_tilde = Y;
                
                % minimizing Y_tilde before X_tilde allows restarts at k=1 instead of k=2 - therefore reduce runtime, without significant (or any) performance change.
                % to reproduce the exact auto-tuned IDBP-BM3D results as in the paper: compute X_tilde before Y_tilde, and set first_iter_to_check=2
                first_iter_to_check = 1; % change to 2 if X_tilde is estimated before Y_tilde
                
                for k=1:1:maxIter
                    
                    % estimate Y_tilde
                    H_conv_X_tilde = cconv2_by_fft2(X_tilde,H,0,[]);
                    Y_tilde = cconv2_by_fft2(Y-H_conv_X_tilde,H,1,epsilon_sig2_e) + X_tilde;
                    
                    % estimate X_tilde
                    % note that "Denoising_operation" is written for treating the denoiser as a "black box"
                    % and not for the fastest performance (e.g. it may load the same DNN and copy between CPU and GPU in each iteration)
                    X_tilde = Denoising_operation(Y_tilde,sigma_alg,denoiser_choice);
                    if max(X_tilde(:))<=1; X_tilde = X_tilde*255; end;
                                       
                    % checking Proposition 1
                    temp1 = Y-cconv2_by_fft2(X_tilde,H,0,[]);
                    temp2 = cconv2_by_fft2(temp1,H,1,epsilon_sig2_e); % apply approximated H^dagger
                    Prop1_LHS = temp1(:)'*temp1(:)/sig_e^2;
                    Prop1_RHS = temp2(:)'*temp2(:)/(sigma_alg)^2;                   
                    
                    if k>=first_iter_to_check && Prop1_LHS/Prop1_RHS < ratio_thr
                        epsilon = epsilon + epsilon_inc;
                        epsilon_sig2_e = epsilon*sig_e^2;
                        epsilon_sig2_e = max([epsilon_sig2_e, 5e-4]);  % protection against sig_e near 0
                        delta = delta + delta_inc;
                        sigma_alg = sig_e + delta; % kept fix for delta_inc = 0
                        disp(['At k=' num2str(k) ': Restarting IDBP with delta = ' num2str(delta) ', epsilon =' num2str(epsilon) ', LHS/RHS ratio = ' num2str(Prop1_LHS/Prop1_RHS) ', old delta=' num2str(delta-delta_inc) ', old epsilon=' num2str(epsilon-epsilon_inc)]);
                        break;
                    elseif k==maxIter
                        flag_IDBP_finished = 1;
                    end

                    if k>1
                        % compute PSNR and ISNR
                        Y_clip = Y; Y_clip(Y<0) = 0; Y_clip(Y>255) = 255;
                        X_tilde_clip = X_tilde; X_tilde_clip(X_tilde<0) = 0; X_tilde_clip(X_tilde>255) = 255;
                        ISNR = 20*log10( norm(X0 - Y_clip, 'fro') / norm(X0 - X_tilde_clip, 'fro'));
                        PSNR = 10*log10(255^2/mean((X0(:)-X_tilde_clip(:)).^2));
                        disp(['IDBP: finished iteration ' num2str(k) ', PSNR for X_tilde = ' num2str(PSNR) ', ISNR = ' num2str(ISNR) ', LHS/RHS ratio = ' num2str(Prop1_LHS/Prop1_RHS)]);
                    end
                end
            end
            
        end        
        
        %% collect results
        
        all_results_ISNR(scenario_ind,image_ind) = ISNR; % you can save PSNR instead
        ssim_res = ssim(double(X_tilde_clip)/255,double(X0)/255); % we use MATLAB R2016a function
        all_results_ssim(scenario_ind,image_ind) = ssim_res;
        disp(['scenario_ind=' num2str(scenario_ind) ', image_ind=' num2str(image_ind) ', ISNR=' num2str(ISNR) ', SSIM=' num2str(ssim_res)]);

    end
end


