function X_tilde = Denoising_operation(Y_tilde,sigma,choice)
% In this function you can add your preferred denoiser.
% We add support for BM3D and IRCNN denoisers, which were used in our TIP paper.
% To use BM3D denoiser, download it from "http://www.cs.tut.fi/foi/GCF-BM3D/BM3D.zip" and unzip it in "Denoisers_folder".
% To use IRCNN denoiser, download it from "https://github.com/cszn/IRCNN" and put it in "Denoisers_folder". However, note that MatConvNet package is required.
% If you use such an off-the-shelf denoiser, please cite also its associated paper, and not only our IDBP paper.

%%% IMPORTANT:
% Note that we wrote this function to allow IDBP treat the denoiser as a "black box"
% and not for the fastest performance (e.g. it may load the same DNN and/or transfer data between CPU and GPU in each iteration).
% For faster performance remove data transfers outside of the IDBP loop.

switch choice
    case 'BM3D'
        [~, X_tilde] = BM3D(0, Y_tilde, sigma, 'np', 0);
        
    case 'IRCNN'
        useGPU = 1; % disable it if you do not have GPU support for MatConvNet
        load('Denoisers_folder\IRCNN\models\modelgray.mat'); % loads CNNdenoiser

        net = loadmodel(sigma,CNNdenoiser);
        net = vl_simplenn_tidy(net);
        if useGPU
            Y_tilde = gpuArray(Y_tilde);
            net = vl_simplenn_move(net, 'gpu');
        end
        res = vl_simplenn(net,single(Y_tilde/255),[],[],'conserveMemory',true,'mode','test');
        X_tilde = Y_tilde - 255*res(end).x;
        if useGPU
            X_tilde = gather(X_tilde);
        end
end



