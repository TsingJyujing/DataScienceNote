clear
clc
load('mnist_number.mat')
Mode = {'Normal','Centeral','ZCA','PCA'};
ModeID = 4;

if ModeID == 3
    [ train_x,Mu,Lambda,W ] = ZCA( train_x,0 );
    [ test_x ] = appZCA( test_x,0,Mu,Lambda,W );
    disp('ZCA Mode')
elseif ModeID == 2
    [ train_x,Mu,Lambda ] = Centeral( train_x,0 );
    [ test_x ] = appCenteral( test_x,0,Mu,Lambda );
    disp('Centeral Mode')
elseif ModeID == 4
    [u,v]  = pca(train_x');
    train_x = v';
    test_x = u'*test_x;
    disp('PCA Mode')
else
    disp('Normal Mode')
end

train_y = one_hot_coder(train_y);
test_tmp_y = test_y;
test_y = one_hot_coder(test_y);

net=newff(minmax(train_x),[50,50,50,10],{'tansig','tansig','tansig','purelin'},'trainscg');
%新建一个神经网络
net.init();%初始化神经网络
net.trainParam.epochs=300; %训练次数设置
net.trainParam.goal=1e-5; %训练所要达到的精度
%net.trainParam.lr=0.01;   %学习速率
net.trainParam.min_grad=1e-8;%最小梯度

try 
    u = gpuDevice;
    net=train(net,train_x,train_y,'useGPU','yes');%开始训练
catch
    net=train(net,train_x,train_y);%开始训练
end

test_res=sim(net,test_x);%测试
[~,out_y] = max(test_res);
out_y = out_y-1;
acc = sum(out_y(:)==test_tmp_y(:));
dif = acc./length(out_y)*100;
disp(['Accuracy:',num2str(dif),'%'])
save('net.mat','net')
