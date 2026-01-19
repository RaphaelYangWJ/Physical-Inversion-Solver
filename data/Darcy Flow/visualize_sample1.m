% ---------- 可视化HDF5文件中第一个样本的结果 ----------
clear; clc; close all;

% HDF5文件名
h5_filename = 'C_H_all_full_field_results.h5';

% 检查文件是否存在
if ~exist(h5_filename, 'file')
    error('HDF5文件不存在: %s', h5_filename);
end

fprintf('正在从HDF5文件读取第一个样本数据...\n');

% 从HDF5文件读取第一个样本的数据
% 浓度数据: 64x64x20，提取第一个样本 (:,:,:,1)
concentration_sample1 = h5read(h5_filename, '/concentration_data', [1,1,1,1], [64,64,20,1]);

% 水头数据: 64x64，提取第一个样本 (:,:,1)  
head_sample1 = h5read(h5_filename, '/head_data', [1,1,1], [64,64,1]);

% 读取时间步信息
time_steps = h5read(h5_filename, '/time_steps');

% 组合成64x64x21的数据结构
field_data = zeros(64, 64, 21);
field_data(:, :, 1:20) = concentration_sample1;
field_data(:, :, 21) = head_sample1;

fprintf('数据读取完成！数据维度: %d x %d x %d\n', size(field_data));

% 创建图形窗口
figure('Position', [100, 100, 1400, 800]);

% 绘制20个浓度时间步
for i = 1:20
    subplot(4, 6, i);
    imagesc(field_data(:, :, i));
    colorbar;
    title(sprintf('浓度 t=%d天', time_steps(i)));
    axis equal tight;
    colormap(gca, 'jet');
    
    % 显示数值范围
    caxis([min(field_data(:, :, i), [], 'all'), max(field_data(:, :, i), [], 'all')]);
end

% 绘制水头数据
subplot(4, 6, 21);
imagesc(field_data(:, :, 21));
colorbar;
title('水头分布');
axis equal tight;
colormap(gca, 'parula');

% 显示数值范围
caxis([min(field_data(:, :, 21), [], 'all'), max(field_data(:, :, 21), [], 'all')]);

% 调整整体标题
sgtitle('HDF5 Sample 1: 浓度演化 (1-20) 和水头分布 (21)', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, 'hdf5_sample1_visualization.png');
saveas(gcf, 'hdf5_sample1_visualization.fig');

fprintf('可视化完成！已保存为 hdf5_sample1_visualization.png 和 .fig 文件\n');

% 显示数据统计信息
fprintf('\n=== 第一个样本数据统计 ===\n');
fprintf('浓度数据范围: %.5f ~ %.5f\n', min(concentration_sample1, [], 'all'), max(concentration_sample1, [], 'all'));
fprintf('水头数据范围: %.5f ~ %.5f\n', min(head_sample1, [], 'all'), max(head_sample1, [], 'all'));
fprintf('时间步范围: %d ~ %d 天\n', min(time_steps), max(time_steps));

% 显示HDF5文件信息
fprintf('\n=== HDF5文件信息 ===\n');
h5disp(h5_filename);