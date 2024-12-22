clear;
clc;

data_path = fullfile(pwd, filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, filesep, "lib", filesep);
addpath(lib_path);
dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
exp_n = 'SMKCDP_test';


for i1 = 1:length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    disp(data_name);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['Create dir: ', dir_name, ' failed, check the authorization']);
    end

    clear X y Y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
    assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));

    fname2 = fullfile(prefix_mdcs, [data_name, '_12kAllFea_', exp_n, '.mat']);
    if ~exist(fname2, 'file')
        Xs = cell(1, 1);
        Xs{1} = X;
        Ks = Xs_to_Ks_12k(Xs);
        Ks2 = Ks{1, 1};
        Ks = Ks2;
        nKernel = size(Ks, 3);
        Ks_cell = cell(1, nKernel);
        for iKernel = 1:nKernel
            Ks_cell{iKernel} = Ks(:, :, iKernel);
        end
        clear Ks Ks2;
        nRepeat = 10;
        seed = 2024;
        rng(seed);
        % Generate 50 random seeds
        random_seeds = randi([0, 1000000], 1, nRepeat);
        % Store the original state of the random number generator
        original_rng_state = rng;
        lambdas = 10.^(-5:1:5);
        paramCell = SMKCDP_MKKM_build_param(lambdas);
        nParam = length(paramCell);
        SMKCDP_MKKM_result = zeros(nParam, 1, nRepeat, 13);  
        SMKCDP_MKKM_time = zeros(nParam, 1);
        for iParam = 1:nParam
            disp(['SMKCDP_MKKM iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
            fname3 = fullfile(prefix_mdcs, [data_name, '_12kAllFea_SMKCDP_MKKM_', num2str(iParam), '.mat']);
            if exist(fname3, 'file')
                load(fname3);
            else
                param = paramCell{iParam};
                NITER = 30;
                t1 = tic;
                for iRepeat = 1:nRepeat
                    % Restore the original state of the random number generator
                    rng(original_rng_state);
                    % Set the seed for the current iteration
                    rng(random_seeds(iRepeat));
                    [Y_EXP, obj_value] = SMKCDP(Ks_cell, nCluster, param.lambda, NITER);
                    Y_normalized = bsxfun(@rdivide, Y_EXP, sqrt(sum(Y_EXP.*Y_EXP, 2)) + 1e-10);
                    label = litekmeans(Y_normalized, nCluster, 'MaxIter', 100, 'Replicates', 10);
                    result_10 = my_eval_y(label, Y);
                   SMKCDP_MKKM_result(iParam, 1, iRepeat, :) = [result_10'];
                end
                tt = toc(t1);
               SMKCDP_MKKM_time(iParam) = tt / nRepeat;
                save(fname3, 'SMKCDP_MKKM_result', 'tt', 'param');
            end
        end

        a1 = sum(SMKCDP_MKKM_result, 2);
        a2 = sum(a1, 3);
        a3 = reshape(a2, nParam, 13);
        a4 = a3 / nRepeat;
        SMKCDP_MKKM_result_summary = [max(a4, [], 1), sum(SMKCDP_MKKM_time) / nParam];
        save(fname2, 'SMKCDP_MKKM_result', 'SMKCDP_MKKM_time', 'SMKCDP_MKKM_result_summary');
        disp([data_name(1:end-4), ' has been completed!']);
    end
end

rmpath(data_path);
rmpath(lib_path);
