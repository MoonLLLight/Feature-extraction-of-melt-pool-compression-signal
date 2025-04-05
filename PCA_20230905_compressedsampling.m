
clc
close all
%%%%%%导入数据
load W300_800_1_9600_clean
L300_800_1 = W300_800_1_9600_clean;
load W300_800_2_9600_clean
L300_800_2 = W300_800_2_9600_clean;
load W300_800_3_9600_clean
L300_800_3 = W300_800_3_9600_clean;
%输出成2D-matrix
L300_800 = [L300_800_1;L300_800_2;L300_800_3];
L300_800_4096 = L300_800(1:1500,1:4096);%800组每组4096
X = L300_800_4096(100:700,1:500);
%%%%%%%%%%%%%%%%%%%%%%
X = (X-min(min(X)))./(max(max(X))-min(min(X)));
%X = X./(max(max(X)));
data = X;
%%%%%%%% 作压缩采样变换，采样率为0
Phi=randn(500,500); 
data = Phi*data.';
data=data';
%%%%%%%%
data1 = randn(100,8);%行数表示维度，每一列代表一组变量,100组每组8个值？
%[a,b]=size(data);
[e,b]=size(data);
[coeff,score,latent] = pca(data);%PCA,压缩列
Var=zeros(1,10);
% coeff 每列对应一个主成分，是否可以认为是特征向量
% score 主成分分数,是 X 在主成分空间中的表示,score 的行对应于观测值，列对应于成分。
% [coeff, score, latent, tsquared, explained, mu]=pca(feature);
a=cumsum(latent)./sum(latent);   % 计算特征的累计贡献率
a=1-a;
% explained和latent均可用来计算降维后取多少维度能够达到自己需要的精度，且效果等价。
% explained=100*latent./sum(latent); 
idx=find(a>0.9);  % 将特征的累计贡献率不小于0.9的维数作为PCA降维后特征的个数
k=idx(1);
Feature=score(:,1:k);   % 取转换后的矩阵score的前k列为PCA降维后特征
for i=2:3
new_data = data*coeff(:,1:i);% coeff为投影矩阵
errorx=sum(sum(abs(new_data-data).^2)); %
Total_variation=sum(sum(abs(new_data).^2));
Var(1,i)=errorx/Total_variation;


end
