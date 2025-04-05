
function DCT_OMP

clc;clear

%%  CS测量
%  读文件
X=imread('lena256.bmp');
X=double(X);
%X = rgb2gray(X);
%  测量矩阵生成(已知)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LoF
% load W300_1_9600_clean
% L300_1 = W300_1_9600_clean;
% load W300_2_9600_clean
% L300_2 = W300_2_9600_clean;
% load W300_3_9600_clean
% L300_3 = W300_3_9600_clean;
% %输出成2D-matrix
% L300 = [L300_1;L300_2;L300_3];
% L300_4096 = L300(1:1350,1:4096);%800组每组4096
% X = L300_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normal
% load W340_1_9600_clean
% L340_1 = W340_1_9600_clean;
% load W340_2_9600_clean
% L340_2 = W340_2_9600_clean;
% load W340_3_9600_clean
% L340_3 = W340_3_9600_clean;
% %输出成2D-matrix
% L340 = [L340_1;L340_2;L340_3];
% L340_4096 = L340(1:1050,1:4096);%800组每组4096
% X = L340_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Keyholes
load W300_800_1_9600_clean
L300_800_1 = W300_800_1_9600_clean;
load W300_800_2_9600_clean
L300_800_2 = W300_800_2_9600_clean;
load W300_800_3_9600_clean
L300_800_3 = W300_800_3_9600_clean;
%输出成2D-matrix
L300_800 = [L300_800_1;L300_800_2;L300_800_3];
L300_800_4096 = L300_800(1:1500,1:4096);%800组每组4096
X = L300_800_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%
X = (X-min(min(X)))./(max(max(X))-min(min(X)));
%X = X./(max(max(X)));
X = X.*255;
X=X';
%%%%%
[a,b]=size(X);
M=26;

%构造高斯测量矩阵，用以随机采样
Phi = randn(M,a); 
Y=Phi*X;
%%  CS恢复
%  DCT变换矩阵生成
ww=DWT(a);%DWT变换矩阵
%ww=dct2(a);
%ww = dctmtx(size(X,1));%DCT变换矩阵
%  测量值(OMP使用)
Y=Y*ww';
%  测量矩阵(OMP使用)
R=Phi*ww';
D=R./repmat(sqrt(sum(R.^2)),[size(R,1) 1]);%归一化，主要针对稀疏变换字典矩阵作归一，而不是测量矩阵，有何区别呢？
%  OMP算法
X2=zeros(a,b);  %  恢复矩阵
param.L=M; % not more than 10 non-zeros coefficients
param.eps=0.0001; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine
for i=1:b  %  列循环       
    rec=mexOMP(Y(:,i),D,param);
    X2(:,i)=rec;
end

%%  CS结果显示

%  原始图像
figure(1);
imshow(uint8(X));
title('原始图像');

%  变换图像
figure(2);
imshow(uint8(X2));
title('恢复离散域的图像');

%  压缩传感恢复的图像

figure(3);
X3=ww'*sparse(X2)*ww;  %  反变换
X3=full(X3*0.071);
imshow(uint8(X3));
title('恢复的空域图像');

%  误差(PSNR)
errorx=sum(sum(abs(X3-X).^2));        %  MSE误差

psnr=10*log10(255*255/(errorx/a/b));   %  PSNR,Signal-to-Noise Ratio (PSNR)