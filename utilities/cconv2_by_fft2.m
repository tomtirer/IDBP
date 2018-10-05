function C = cconv2_by_fft2(A,B,flag_invertB,epsilon)
% assumes that A (image) is bigger than B (kernel)

[m,n] = size(A);
[mb,nb] = size(B);

% pad, multiply and transform back

bigB = zeros(m,n); bigB(1:mb,1:nb)=B; bigB=circshift(bigB,-round([(mb-1)/2 (nb-1)/2])); % pad PSF with zeros to whole image domain, and center it

fft2B = fft2(bigB);
if flag_invertB
    fft2B = conj(fft2B)./( (abs(fft2B).^2) + epsilon); % Standard Tikhonov Regularization
end

C = ifft2(fft2(A).* fft2B);

