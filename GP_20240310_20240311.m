%% 对应大论文6-4节
%% 导入训练集数据
categories = {'Keyholes','LoF','Normal'};%只分Keyhole和Normal准确率很高
CurrentPath = cd;% 获取当前文件夹路径
index_dir = strfind(CurrentPath,'\'); %寻找路径中的"\"
str_temp = CurrentPath(1:index_dir(end)-1);
rootFolder = [str_temp '\CompressedRatio75\R40\cifar10Train'];%%CompressedRatio25
imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource','foldernames'); 
[trainingSet,validationSet,testSet] = splitEachLabel(imds,2000,400,300,'randomize');%7200*0.8/3=1920(1536,384)组训练，100组验证
%% 分割训练和验证集 
%% 获取训练集图像trainImages和标签trainLabels
trainImageFiles = trainingSet.Files;
trainLabels = trainingSet.Labels;
% 读取训练集中的图像
% trainImages = cell(length(trainImageFiles), 1);
% for i = 1:length(trainImageFiles)
%     trainImages{i} = imread(trainImageFiles{i});
% end
%% 获取测试集图像testImages和标签testLabels
testImageFiles = validationSet.Files;
testLabels = validationSet.Labels;
% 读取训练集中的图像
testImages = cell(length(testImageFiles), 1);
% for i = 1:length(testImageFiles)
%     testImages{i} = imread(testImageFiles{i});
% end
%% 

% 导入 ResNet-50 模型
load('Pretrain_net_compress_75_80_20240311.mat');
% 获取倒数第二个残差块的名称
layerName = 'fc_1';

% 提取倒数第二个残差块的输出作为特征
XTrain=trainImageFiles;
XTest=testImageFiles;

[trainFeatures,TrainLabels] = GP_pre_20240310_20240311(net, XTrain, layerName,trainLabels);
[testFeatures,TestLabels] = GP_pre_20240310_20240311(net, XTest, layerName,testLabels);
%%%%%%%%%%%%

% 使用高斯过程回归作为分类器
YTest=testLabels;
YTestIndices = grp2idx(YTest); % 将分类变量转换为数值索引
YTestDouble = double(YTestIndices); % 将数值索引转换为双精度向量
 gp = fitrgp(trainFeatures, TrainLabels);
YTest = TestLabels;
% 使用测试集测试高斯过程分类器性能
YPredGP = predict(gp, testFeatures);
YPredInteger = round(YPredGP); % 四舍五入到最接近的整数
% 计算正确预测的数量
correctPredictions = sum(YPredInteger == YTest);
% 计算总样本数量
totalSamples = numel(YTest);
% 计算准确率
accuracy = (correctPredictions / totalSamples) * 100;
                                   
