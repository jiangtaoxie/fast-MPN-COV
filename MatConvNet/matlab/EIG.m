function [V,S,D] = EIG(x)
% EIG 
% eigenvalue decomposition of a batch of covariance matrices
% created by Jiangtao Xie for MPN-COV
type = class(x);
L = size(x,3);
n = size(x,1);
V = zeros(n,n,L,type);
S = zeros(n,1,L,type);
D = zeros(L,1,type);
idx = n : -1 : 1;
for i = 1 : L
    [v,diag_S] = eig(x(:,:,i),'vector');
    diag_S = diag_S(idx);
    V(:,:,i) = v(:, idx);
    ind    = diag_S  > ( eps(max(diag_S))); 
    Dmin = min(find(ind, 1, 'last'), n);
    D(i) = Dmin;
    V(:, Dmin + 1 : end,i) = 0; 
    diag_S(Dmin + 1 : end) = 0;
    S(:,:,i) = diag_S;
end
end



