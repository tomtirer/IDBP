function [x_star, iter, residual] = cg(x0, A, y, i_max, tau)
% Solve a symmetric positive definite system Ax = y via conjugate gradients.
% A is a function handler

x = x0;
r = y - A(x0);
d = r;
delta = r'*r;

for k=0:1:i_max
    if delta<tau^2
        break;
    end
    q = A(d);
    alpha = delta/(d'*q);
    x = x + alpha*d;
    r = r - alpha*q;
    deltaold = delta;
    delta = r'*r;
    beta = delta/deltaold;
    d = r + beta*d;
%     if mod(k,10)==0
%        disp(['cg: finished iteration k = ' num2str(k) ', norm(residual) = ' num2str(sqrt(delta))]);
%     end
end

residual = delta;
x_star = x;
iter = k;

