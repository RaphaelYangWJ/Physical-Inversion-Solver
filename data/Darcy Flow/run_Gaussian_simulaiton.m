clear; clc; close all;
rng('shuffle');

currentdir = pwd;
addpath([currentdir,'\Utilities']);
gcp;

% ---------- 参数设置 ----------
Ne       = 200;
Lx       = 63;
Ly       = 63;
lambdax  = 30;
lambday  = 15;
MeanY    = 2.0;
VarY     = 0.5;
kl_num   = 400;

t        = 50:50:1000;                  % 20 个应力期末
timestep = numel(t);

% ---------- 拷贝算例 ----------
exampledir = fullfile(currentdir, 'high_fidelity');
cd(exampledir);
copyexample(Ne);
cd(currentdir);

% ---------- 生成 KL 系数矩阵 ----------
kl_terms_matrix = generate_kl_random_field( ...
    Lx, Ly, lambdax, lambday, MeanY, VarY, kl_num, Ne);

S   = load('fun.mat', 'fun');               % fun: 生成 Y 的基函数矩阵
fun = S.fun;

% ---------- 预分配输出 ----------
% 现在每个样本返回64*64*21的三维数组，需要用cell数组或4维数组存储
full_field_results = zeros(64, 64, timestep + 1, Ne);  % 64*64*21*Ne

% ---------- 批量模拟 ----------
tic
parfor i = 1:Ne
    full_field_results(:, :, :, i) = model_H( ...
        kl_terms_matrix(:, i), ...        % 本次样本的 KL 系数
        i, ...                            % 样本序号，用于并行子目录
        MeanY, ...
        fun, ...
        t, ...
        exampledir);
end
toc

% ---------- 保存完整场域数据 ----------
fprintf('正在保存数据，数据大小: %.2f GB\n', numel(full_field_results)*8/1024^3);

% 使用HDF5格式保存，数据精确到小数点后五位
tic;
h5_filename = 'C_H_all_full_field_results.h5';
if exist(h5_filename, 'file')
    delete(h5_filename);
end

% 将数据四舍五入到小数点后五位，然后转换为single精度
concentration_data = single(round(full_field_results(:, :, 1:20, :), 5));
% 注意：需要squeeze掉单维度，将64x64x1xNe转换为64x64xNe
head_data = single(round(squeeze(full_field_results(:, :, 21, :)), 5));

% 创建HDF5文件并写入数据，使用压缩
h5create(h5_filename, '/concentration_data', [64, 64, 20, Ne], 'Datatype', 'single', 'ChunkSize', [64, 64, 1, 1], 'Deflate', 6);
h5create(h5_filename, '/head_data', [64, 64, Ne], 'Datatype', 'single', 'ChunkSize', [64, 64, 1], 'Deflate', 6);
h5create(h5_filename, '/time_steps', [20, 1], 'Datatype', 'double');

% 写入精确到小数点后五位的数据
h5write(h5_filename, '/concentration_data', concentration_data);
h5write(h5_filename, '/head_data', head_data);
h5write(h5_filename, '/time_steps', t');

% 添加属性信息
h5writeatt(h5_filename, '/', 'description', 'Full field simulation results (precision: 5 decimal places)');
h5writeatt(h5_filename, '/', 'dimensions', 'concentration: [64x64x20xNe], head: [64x64xNe]');
h5writeatt(h5_filename, '/', 'precision', 'Data rounded to 5 decimal places');
h5writeatt(h5_filename, '/', 'Ne', Ne);
h5writeatt(h5_filename, '/concentration_data', 'units', 'concentration');
h5writeatt(h5_filename, '/head_data', 'units', 'head');

save_time_h5 = toc;
h5_info = dir(h5_filename);
fprintf('HDF5保存完成！\n');
fprintf('  用时: %.2f 秒\n', save_time_h5);
fprintf('  文件大小: %.2f GB\n', h5_info.bytes/1024^3);
fprintf('  数据精度: 小数点后5位\n');

% 清理内存
clear full_field_results concentration_data head_data;

% Delete the files for parallel computation
cd([currentdir,'\high_fidelity'])
copyexample(Ne,-1);        %Zx 删除
cd(currentdir);