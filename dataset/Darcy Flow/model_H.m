function yout = model_H(kl_term, ii, MeanY, fun, t, exampledir)
maindir = pwd;
parpath = fullfile(exampledir, sprintf('parallel_%d', ii));
trandat = fullfile(parpath, 'Tran.dat');

timestep = numel(t);

% 预分配完整场域数据：64*64*21 (20个浓度时间步 + 1个水头)
full_field_data = zeros(64, 64, timestep + 1);

% ---------- 生成并写入 K 场 ----------
Y = MeanY + fun*kl_term;
Kcond   = exp(Y);
K_field = reshape(Kcond,64,64)';   % 与原版一致
Tran    = reshape(K_field', [], 1);  % 与原版一致（多次检验通过的写法）
dlmwrite(trandat, Tran, 'delimiter','', 'precision','%10.4f','newline','pc');

% ---------- 运行批处理 ----------

cd(parpath);
system('mt3dms5b.bat');
cd(maindir);

% ---------- 读取浓度 UCN ----------
UCNfile = fullfile(parpath, 'MT3D001.UCN');
CC      = readMT3D(UCNfile);           % 期望返回 struct 数组，含 .time 和 .values

% 定位所需时间切片
for i = 1:size(CC, 1)
    Ctime(i) = CC(i).time;                      
end
[m, ~] = find(Ctime' == t);

for j = 1:timestep
    tpcon = CC(m(j)).values;
    full_field_data(:, :, j) = tpcon;  % 保存完整的64*64浓度场
end

% ---------- 读取水头 HED ----------
HEDfile = fullfile(parpath, 'zx_7_12.hed');
H       = readDat(HEDfile);            % 期望返回 struct，含 .values (矩阵)

tphead = H.values;
full_field_data(:, :, timestep + 1) = tphead;  % 保存完整的64*64水头场

% ---------- 拼装输出 ----------
yout = full_field_data;  % 返回64*64*21的完整场域数据

end

