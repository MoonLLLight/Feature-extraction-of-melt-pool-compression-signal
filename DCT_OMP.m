
function DCT_OMP

clc;clear

%%  CS����
%  ���ļ�
X=imread('lena256.bmp');
X=double(X);
%X = rgb2gray(X);
%  ������������(��֪)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LoF
% load W300_1_9600_clean
% L300_1 = W300_1_9600_clean;
% load W300_2_9600_clean
% L300_2 = W300_2_9600_clean;
% load W300_3_9600_clean
% L300_3 = W300_3_9600_clean;
% %�����2D-matrix
% L300 = [L300_1;L300_2;L300_3];
% L300_4096 = L300(1:1350,1:4096);%800��ÿ��4096
% X = L300_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normal
% load W340_1_9600_clean
% L340_1 = W340_1_9600_clean;
% load W340_2_9600_clean
% L340_2 = W340_2_9600_clean;
% load W340_3_9600_clean
% L340_3 = W340_3_9600_clean;
% %�����2D-matrix
% L340 = [L340_1;L340_2;L340_3];
% L340_4096 = L340(1:1050,1:4096);%800��ÿ��4096
% X = L340_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Keyholes
load W300_800_1_9600_clean
L300_800_1 = W300_800_1_9600_clean;
load W300_800_2_9600_clean
L300_800_2 = W300_800_2_9600_clean;
load W300_800_3_9600_clean
L300_800_3 = W300_800_3_9600_clean;
%�����2D-matrix
L300_800 = [L300_800_1;L300_800_2;L300_800_3];
L300_800_4096 = L300_800(1:1500,1:4096);%800��ÿ��4096
X = L300_800_4096(1:256,1:256);
%%%%%%%%%%%%%%%%%%%%%%
X = (X-min(min(X)))./(max(max(X))-min(min(X)));
%X = X./(max(max(X)));
X = X.*255;
X=X';
%%%%%
[a,b]=size(X);
M=26;

%�����˹�������������������
Phi = randn(M,a); 
Y=Phi*X;
%%  CS�ָ�
%  DCT�任��������
ww=DWT(a);%DWT�任����
%ww=dct2(a);
%ww = dctmtx(size(X,1));%DCT�任����
%  ����ֵ(OMPʹ��)
Y=Y*ww';
%  ��������(OMPʹ��)
R=Phi*ww';
D=R./repmat(sqrt(sum(R.^2)),[size(R,1) 1]);%��һ������Ҫ���ϡ��任�ֵ��������һ�������ǲ��������к������أ�
%  OMP�㷨
X2=zeros(a,b);  %  �ָ�����
param.L=M; % not more than 10 non-zeros coefficients
param.eps=0.0001; % squared norm of the residual should be less than 0.1
param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                    % and uses all the cores of the machine
for i=1:b  %  ��ѭ��       
    rec=mexOMP(Y(:,i),D,param);
    X2(:,i)=rec;
end

%%  CS�����ʾ

%  ԭʼͼ��
figure(1);
imshow(uint8(X));
title('ԭʼͼ��');

%  �任ͼ��
figure(2);
imshow(uint8(X2));
title('�ָ���ɢ���ͼ��');

%  ѹ�����лָ���ͼ��

figure(3);
X3=ww'*sparse(X2)*ww;  %  ���任
X3=full(X3*0.071);
imshow(uint8(X3));
title('�ָ��Ŀ���ͼ��');

%  ���(PSNR)
errorx=sum(sum(abs(X3-X).^2));        %  MSE���

psnr=10*log10(255*255/(errorx/a/b));   %  PSNR,Signal-to-Noise Ratio (PSNR)