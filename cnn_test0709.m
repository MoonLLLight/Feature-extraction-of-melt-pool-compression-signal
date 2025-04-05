%% 清空环境变量
clear 
clc
%% 导入数据集并sss.trainset
%导入数据 W220_1000
% load W220_1000_1_double
% L220_1000 = W220_1000_1_double;
% load W220_1000_2_double
% L220_1000_1 = W220_1000_2_double;
% load W220_1000_3_double
% L220_1000_2 = W220_1000_3_double;
% %输出成2D-matrix
% L220_1000 = [L220_1000;L220_1000_1;L220_1000_2];
% L220_1000 = L220_1000(:,1:900);
% SL220_1000 = zeros(900,30,30);
% AL220_1000 = zeros(30,30);
% for i = 1:900
%     AL220_1000 = reshape(L220_1000(i,:,:)',30,30)';
%     AL220_1000 = AL220_1000./max(AL220_1000);
%     SL220_1000(i,:,:) = AL220_1000;% X为序列，每个YZ为按照时序索取水平自上往下后成的30*30矩阵
%     path =[ 'G:\3D打印课题组\individual\在线监测-辐射信号-质量特征评估\信号重构预处理\数据集\Dataset\cifar10Train\SL220_1000\',num2str(i),'.bmp'];
%     imwrite(AL220_1000,path,'bmp');
% end
% 
% %导入数据 W260
% load W260_1_double
% L260 = W260_1_double;
% load W260_2_double
% L260_1 = W260_2_double;
% load W260_3_double
% L260_2 = W260_3_double;
% %输出成图像
% L260 = [L260;L260_1;L260_2];
% L260 = L260(:,1:900);
% SL260 = zeros(900,30,30);
% AL260 = zeros(30,30);
% for i = 1:900
%     AL260 = reshape(L260(i,:,:)',30,30)';
%     AL260 = AL260./max(AL260);
%     SL260(i,:,:) = AL260;% X为序列，每个YZ为按照时序索取水平自上往下后成的30*30矩阵
%     path =[ 'G:\3D打印课题组\individual\在线监测-辐射信号-质量特征评估\信号重构预处理\数据集\Dataset\cifar10Train\SL260\',num2str(i),'.bmp'];
%     imwrite(AL260,path,'bmp');
% end
% 
% %导入数据 W260_1200
% load W260_1200_1_double
% L260_1200 = W260_1200_1_double;
% load W260_1200_2_double
% L260_1200_1 = W260_1200_2_double;
% load W260_1200_3_double
% L260_1200_2 = W260_1200_3_double;
% %输出成图像
% L260_1200 = [L260_1200;L260_1200_1;L260_1200_2];
% L260_1200 = L260_1200(:,1:900);
% SL260_1200 = zeros(900,30,30);
% AL260_1200 = zeros(30,30);
% for i = 1:900
%     AL260_1200 = reshape(L260_1200(i,:,:)',30,30)';
%     AL260_1200 = AL260_1200./max(AL260_1200);
%     SL260_1200(i,:,:) = AL260_1200;% X为序列，每个YZ为按照时序索取水平自上往下后成的30*30矩阵
%     path =[ 'G:\3D打印课题组\individual\在线监测-辐射信号-质量特征评估\信号重构预处理\数据集\Dataset\cifar10Train\SL260_1200\',num2str(i),'.bmp'];
%     imwrite(AL220_1000,path,'bmp');
% end
% %  imwrite(frame,strcat('./视频分解/',num2str(k),'.jpg'),'jpg');
% %%%%
% 
% trainset = [SL220_1000;SL260;SL260_1200];
% save trainset
%% 标签 categories
% categories = zeros(903,3);
% for i=1:301
%     categories(i,1)=1;
% end
% 
% for i=302:602
%     categories(i,2)=1;
% end
% 
% for i=603:903
%     categories(i,3)=1;
% end
% %%怎么把数据集和标签放到一个结构体里呢
%% 导入训练集数据
categories = {'Keyholes','LoF','Normal'};%'SL220_1000','SL260','SL260_1200'
CurrentPath = cd;% 获取当前文件夹路径
index_dir = strfind(CurrentPath,'\'); %寻找路径中的"\"
str_temp = CurrentPath(1:index_dir(end)-1);
rootFolder = [str_temp '\Uncompressed\R32\cifar10Train'];
imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource','foldernames'); 
[trainingSet,validationSet] = splitEachLabel(imds,1536,384,'randomize');%7200*0.8/3=1920(1536,384)组训练，100组验证
%% 定义卷积神经网络结构
varSize1 = 32;% 输入尺寸W
varSize2 = 32;% 输入尺寸H
varSize = 32;%卷积核个数
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);%补两行两列0，BiasLearnRateFactor学习率因子
fc1 = fullyConnectedLayer(36,'BiasLearnRateFactor',2);%全连接层，神经元16
fc2 = fullyConnectedLayer(3,'BiasLearnRateFactor',2);%全连接层，神经元2，与类别标签数一致
layers =[imageInputLayer([varSize1 varSize2 1]);
    conv1;
    batchNormalizationLayer;%批量归一层
    maxPooling2dLayer(3,'stride',2);%最大池化层，尺寸3*3，步长2
    reluLayer();%激活函数层
    convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
    batchNormalizationLayer;
    maxPooling2dLayer(3,'stride',2);%最大池化层，尺寸3*3，步长2
    reluLayer();
    %averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(3,64,'Padding',2,'BiasLearnRateFactor',2,'Stride',2);

    batchNormalizationLayer;
    %%%%%%%  输入最小设置为16时，最大允许的池化尺寸为2*2而不是3*3，但为2的话效果很差
    reluLayer;averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    dropoutLayer(0.2)               % Dropout层
    softmaxLayer() % 给出分配标签的概率，多个标量映射为一个概率分布，其输出的每一个值范围在(0,1)
    classificationLayer()];
%% 设置训练参数
opts = trainingOptions('adam',...%训练算法，优化器
                       'InitialLearnRate',0.001, ...%初始学习率
                       'LearnRateSchedule', 'piecewise',...%学习率不变还是分段线性
                       'LearnRateDropFactor',0.05,...%如为分段线性，隔一段减小学习率0.05
                       'LearnRateDropPeriod',10,...%如为分段线性，间隔为10
                       'L2Regularization',0.004,...%
                       'MaxEpochs',50, ...%最大迭代次数
                       'MiniBatchSize',40,...%分批输入，每批输入样本数
                       'Verbose',false,...
                       'ValidationData',validationSet,...%指定验证集
                       'ValidationFrequency',50,...%验证集，每隔50次迭代验证一次，改为按照每个epoch对应的迭代次数
                       'ExecutionEnvironment','cpu',...
                       'Plot','training-progress'); %绘制迭代训练过程
                        %%'ExecutionEnvironment','cpu',...%%参数设置里默认为GPU，可转CPU
                        %%out1 =predict(net,pic1,'executionEnvironment','cpu');%
                        %调用预测时默认GPU，可转CPU
                        %'ValidationData',validationSet,...%指定验证集
                        %'Shuffle', 'every-epoch', ... % 每次训练打乱数据集
%% 训练卷积神经网络
[net, info]=trainNetwork(trainingSet,layers,opts);%数据集；网络拓扑结构；训练参数
%trainNetwork(train_data.input_data,train_data.output_data,layers,options);
%% 导入测试集数据
CurrentPath = cd;%获取当前文件夹路径
index_dir = strfind(CurrentPath,'\');%寻找路径中的"\"
str_temp = CurrentPath(1:index_dir(end)-1);% 获取上一级文件夹路径
rootFolder = [str_temp '\Uncompressed\R32\cifar10Test'];
imds_test = imageDatastore(fullfile(rootFolder, categories),'LabelSource','foldernames');
[testingSet,~]= splitEachLabel(imds_test,1000,'randomize');
%% 模型预测
labels = classify(net,testingSet);
%labels = classify(net,validationSet);
%% 计算混淆矩阵
confMat = confusionmat(testingSet.Labels,labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
%y=stackednet(xTest)%y为计算值
%plotconfusion(tTest,y)%tTest为真实值
% plot(info.TrainingLoss)；%%画出训练的loss
% plot(info.TrainingAccuracy)；%%画出训练的准确率









