%% 对应大论文6-3节
%% 导入训练集数据
folderPath = 'E:\AA-ZHX\图像超分辨与增强\ImageNet -1k (ILSVRC)数据集\ILSVRC2012_img_train\ImageNet-1k'; % 替换为实际的路径
% 取所有数据集
% imds = imageDatastore(folderPath, 'IncludeSubfolders', true, 'FileExtensions', '.JPEG', 'LabelSource', 'foldernames');
% 只取部分数据集
%%%%    ‘-1’是原始图像直接扩充；‘-resize224’是原始图像缩放到224；‘-resize224-expanded’是对缩放后图像扩充
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

desiredSubfolders = {'n01440764-Resize224-expanded', 'n01443537-Resize224-expanded', 'n01484850-Resize224-expanded',...
    'n01491361-Resize224-expanded','n01494475-Resize224-expanded','n01496331-Resize224-expanded',...
    'n01558993-Resize224-expanded','n01631663-Resize224-expanded','n01514859-Resize224-expanded','n01518878-Resize224-expanded'};
% desiredSubfolders = {'n01440764-resize224', 'n01443537-resize224', 'n01484850-resize224',...
%     'n01491361-resize224','n01494475-resize224','n01496331-resize224',...
%     'n01558993-resize224','n01631663-resize224','n01514859-resize224','n01518878-resize224'};
% desiredSubfolders = { 'n01484850',...
%    'n01491361','n01494475','n01496331','n01514859'};
fullSubfolderPaths = fullfile(folderPath, desiredSubfolders);
imds = imageDatastore(fullSubfolderPaths, 'FileExtensions', '.JPEG', 'LabelSource', 'foldernames');
%% 分割训练和验证集 
% [trainingSet,validationSet] = splitEachLabel(imds,1000,200,'randomize');%
[trainingSet,validationSet] = splitEachLabel(imds,0.8,'randomize');%
% 定义增强参数
augmenter = imageDataAugmenter('RandXReflection',false, ...% 不应用关于x轴的随机反射（即水平翻转）
                               'RandYReflection',false, ...% 不应用关于y轴的随机反射（即垂直翻转）
                               'RandRotation',[0 0], ...% 不应用随机旋转
                               'RandXTranslation',[0 0], ...% 不应用关于x轴随机平移
                               'RandYTranslation',[0 0], ...% 不应用关于y轴随机平移
                               'RandXScale',[1 1], ...% 随机缩放无
                               'RandYScale',[1 1]);% 调整尺寸无
trainingSet_1 = augmentedImageDatastore([448 448], trainingSet, 'DataAugmentation', augmenter,'ColorPreprocessing', 'gray2rgb');
validationSet_1 = augmentedImageDatastore([448 448], validationSet, 'ColorPreprocessing', 'gray2rgb');% 如存在灰度图像则转换为RGB
%% 获取训练集图像trainImages和标签trainLabels
trainImageFiles = trainingSet_1.Files;
trainLabels = trainingSet.Labels;
% 读取训练集中的图像
% trainImages = cell(length(trainImageFiles), 1);
% for i = 1:length(trainImageFiles)
%     trainImages{i} = imread(trainImageFiles{i});
% end
%% 获取测试集图像testImages和标签testLabels
testImageFiles = validationSet_1.Files;
testLabels = validationSet.Labels;
% 读取训练集中的图像
testImages = cell(length(testImageFiles), 1);
% for i = 1:length(testImageFiles)
%     testImages{i} = imread(testImageFiles{i});
% end
%% 

% 导入 ResNet-50 模型

load('ResNet18_ImageNet_10_224_expanded_448_20240306_step4_Gass.mat');
% 获取倒数第二个残差块的名称
layerName = 'pool5';

% 提取倒数第二个残差块的输出作为特征
XTrain=trainImageFiles;
XTest=testImageFiles;

[trainFeatures,TrainLabels] = GP_pre_20240310(net, XTrain, layerName,trainLabels);
[testFeatures,TestLabels] = GP_pre_20240310(net, XTest, layerName,testLabels);
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
                                   
