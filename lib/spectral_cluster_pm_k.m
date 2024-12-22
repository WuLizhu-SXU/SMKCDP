function U = spectral_cluster_pm_k(A, nCluster, t_const)
if nargin < 3
    t_const = 2;
end
nSmp = size(A, 1);
logn = t_const * ceil(log2(nSmp));
eigenvectors = randn(nSmp, nCluster);

for i = 1:logn
    eigenvectors = A * eigenvectors;
end

% Orthogonalize
[U, ~, ~] = svd(eigenvectors, 'econ');
end
