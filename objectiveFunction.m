

function [MAPE,T_sim]= objectiveFunction(x,f_,vp_train,vt_train,vp_test,T_test,ps_output)

%% 将优化目标参数传进来的值 转换为需要的超参数
learning_rate = x(1);            %% 学习率
NumNeurons = round(x(2));        %% biGRU神经元个数
keys = round(x(3));        %% 自注意力机制的键值数
L2Regularization = x(4);   %正则化参数
setdemorandstream(pi);

outputSize = 1;  %数据输出y的维度  
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.1;
numBlocks = 2;

layer = sequenceInputLayer(f_,Normalization="rescale-symmetric",Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor) 
        % spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor) 
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end


tempLayers = flattenLayer("Name","flatten");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = gruLayer(NumNeurons,"Name","gru1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    FlipLayer("flip3")
    gruLayer(NumNeurons,"Name","gru2")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    concatenationLayer(1,2,"Name","concat")
    selfAttentionLayer(1,keys,"Name","selfattention")   %单头注意力Attention机制，把1改为2,3,4……即为多头，后面的50是键值
    fullyConnectedLayer(outputSize,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);


lgraph = connectLayers(lgraph,outputName,"flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");


%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 60, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', learning_rate, ...         % 初始学习率
    'L2Regularization', L2Regularization, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

%% 训练网络
net0 = trainNetwork(vp_train,vt_train,lgraph,options);
% analyzeNetwork(net);% 查看网络结构 
                     
%% 测试与评估
t_sim = net0.predict(vp_test);  

%  数据反归一化
T_sim = mapminmax('reverse', t_sim, ps_output);

%  数据格式转换
T_sim = cell2mat(T_sim);
T_sim = T_sim';
%% 计算误差
MAPE=sum(abs((T_sim-T_test)./T_test))/length(T_test);  % 平均百分比误差
display(['本批次MAPE:', num2str(MAPE)]);
end

