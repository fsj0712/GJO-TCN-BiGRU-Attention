
%% 金豺算法优化TCN-BiGRU-Attention，实现多变量输入单步预测

clc;
clear 
close all

X = xlsread('4.data_2020.xlsx');
X = X(5666:8642,:);  %选取3月份数据
num_samples = length(X);                            % 样本个数 
kim = 10;                      % 延时步长（kim个历史数据作为自变量）
zim =  1;                      % 跨zim个时间点进行预测
or_dim = size(X,2);

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.9;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end



%% 优化算法优化前，构建优化前的TCN_BiGRU_Attention模型

outputSize = 1;  %数据输出y的维度  
numFilters = 128;
filterSize = 5;
dropoutFactor = 0.08;
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

tempLayers = gruLayer(10,"Name","gru1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    FlipLayer("flip3")
    gruLayer(10,"Name","gru2")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    concatenationLayer(1,2,"Name","concat")
    selfAttentionLayer(1,50,"Name","selfattention")   %单头注意力Attention机制，把1改为2,3,4……即为多头，后面的50是键值
    fullyConnectedLayer(outdim,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);


lgraph = connectLayers(lgraph,outputName,"flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");


%  参数设置
options0 = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 60, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'L2Regularization', 0.0001, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

% 网络训练
tic
net0 = trainNetwork(vp_train,vt_train,lgraph,options0);
toc
%% 测试与评估
t_sim = net0.predict(vp_test);  
analyzeNetwork(net0);% 查看网络结构

%  数据反归一化
T_sim = mapminmax('reverse', t_sim, ps_output);

%  数据格式转换
T_sim = cell2mat(T_sim);
T_sim = T_sim';


%% GJO优化TCN-BiGRU-Attention

disp(' ')
disp('GJO优化TCN_BiGRU_attention神经网络：')

%% 初始化GJO参数 
popsize=5;   %初始种群规模 
maxgen=10;   %最大进化代数
fobj = @(x)objectiveFunction(x,f_,vp_train,vt_train,vp_test,T_test,ps_output);
% SSA优化参数设置
lb = [0.001 10 2  0.0001]; %参数的下限。分别是学习率，biGRU的神经元个数，注意力机制的键值, 正则化参数
ub = [0.01 50 50 0.001];    %参数的上限
dim = length(lb);%数量

[Best_score,Best_pos,GJO_curve]=GJO(popsize,maxgen,lb,ub,dim,fobj);
setdemorandstream(pi);

%% 绘制进化曲线 
figure
plot(GJO_curve,'r-','linewidth',2)
xlabel('Generation')
ylabel('MSE')
legend('optiinal fitness ')
title('learning curve of GJO')

%% 把最佳参数Best_pos回带 
[~,optimize_T_sim] = objectiveFunction(Best_pos,f_,vp_train,vt_train,vp_test,T_test,ps_output);
setdemorandstream(pi);

%% 比较算法预测值 
str={'the real data','TCN-BiGRU-Attention','optimised TCN-BiGRU-Attention'};
figure('Units', 'pixels', ...
    'Position', [300 300 860 370]);
plot(T_test,'-','Color',[0.8500 0.3250 0.0980]) 
hold on
plot(T_sim,'-.','Color',[0.4940 0.1840 0.5560]) 
hold on
plot(optimize_T_sim,'-','Color',[0.4660 0.6740 0.1880])
legend(str)
set (gca,"FontSize",12,'LineWidth',1.2)
box off
legend Box off


%% 比较算法误差
test_y = T_test;
Test_all = [];

y_test_predict = T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];


y_test_predict = optimize_T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];
 	

str={'the real data','TCN-BiGRU-Attention','optimised TCN-BiGRU-Attention'};
str1=str(2:end);
str2={'MAE','MAPE','MSE','RMSE','R2'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)

%% 柱状图 MAE MAPE RMSE 柱状图适合量纲差别不大的
color=    [0.66669    0.1206    0.108
    0.1339    0.7882    0.8588
    0.1525    0.6645    0.1290
    0.8549    0.9373    0.8275   
    0.1551    0.2176    0.8627
    0.7843    0.1412    0.1373
    0.2000    0.9213    0.8176
      0.5569    0.8118    0.7882
       1.0000    0.5333    0.5176];
figure('Units', 'pixels', ...
    'Position', [300 300 660 375]);
plot_data_t=Test_all(:,[1,2,4])';
b=bar(plot_data_t,0.8);
hold on

for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor=[0.3353    0.3314    0.6431];
    b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=xline(xilnk,'--','LineWidth',1.2);
    hold on
end 

ax=gca;
legend(b,str1,'Location','best')
ax.XTickLabels ={'MAE', 'MAPE', 'RMSE'};
set(gca,"FontSize",10,"LineWidth",1)
box off
legend box off

%% 二维图
figure
plot_data_t1=Test_all(:,[1,5])';
MarkerType={'s','o','pentagram','^','v'};
for i = 1 : size(plot_data_t1,2)
   scatter(plot_data_t1(1,i),plot_data_t1(2,i),120,MarkerType{i},"filled")
   hold on
end
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
legend(str1,'Location','best')
xlabel('MAE')
ylabel('R2')
grid on


%% 雷达图
figure('Units', 'pixels', ...
    'Position', [150 150 520 500]);
Test_all1=Test_all./sum(Test_all);  %把各个指标归一化到一个量纲
Test_all1(:,end)=1-Test_all(:,end);
RC=radarChart(Test_all1);
str3={'MAE','MAPE','MSE','RMSE','R2'};
RC.PropName=str3;
RC.ClassName=str1;
RC=RC.draw(); 
RC.legend();
RC.setBkg('FaceColor',[1,1,1])
RC.setRLabel('Color','none')
colorList=[78 101 155;
          181 86 29;
          184 168 207;
          231 188 198;
          253 207 158;
          239 164 132;
          182 118 108]./255;

for n=1:RC.ClassNum
    RC.setPatchN(n,'Color',colorList(n,:),'MarkerFaceColor',colorList(n,:))
end



%%
figure('Units', 'pixels', ...
    'Position', [150 150 920 600]);
t = tiledlayout('flow','TileSpacing','compact');
for i=1:length(Test_all(:,1))
nexttile
th1 = linspace(2*pi/length(Test_all(:,1))/2,2*pi-2*pi/length(Test_all(:,1))/2,length(Test_all(:,1)));
r1 = Test_all(:,i)';
[u1,v1] = pol2cart(th1,r1);
M=compass(u1,v1);
for j=1:length(Test_all(:,1))
    M(j).LineWidth = 2;
    M(j).Color = colorList(j,:);

end   
title(str2{i})
set(gca,"FontSize",10,"LineWidth",1)
end
 legend(M,str1,"FontSize",10,"LineWidth",1,'Box','off','Location','southoutside')




