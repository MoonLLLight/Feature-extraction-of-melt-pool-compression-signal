function [features, labels] = GP_pre_20240310_20240311(net,images,layerName,Labels)
    num_images = numel(images);
    features = zeros(num_images, 128); % ResNet-18 最后一个全连接层的输出为 512 维
     % 创建用于保存对应标签的数组
    labels = zeros(numel(images), 1);
    for i = 1:num_images

        img = imread(images{i});

        % 如果图像是三通道的 RGB 图像，则保存对应标签
        labels(i) = Labels(i); % 将对应标签保存到 labels 数组中

%    %% 检查图像是否为三通道的 RGB 图像
%     if size(img, 3) ~= 3
%         % 如果图像不是三通道的 RGB 图像，则转换为三通道
%         img = cat(3, img, img, img); % 将单通道图像复制为三通道图像
%     end


        % 提取 ResNet 模型最后一个全连接层的输出
        activation = activations(net, img, layerName);
        feature = activation(:);
        
        % 将提取的特征保存到特征矩阵中
        features(i, :) = feature;
    end

end