clear;

clc;close all;
load('Datapack.mat')

%parameters setting
thread_count = 20;
batchN = 1;
alpha = 0.5;

%prepare pointers for dll(*.so file)
pData = libpointer('doublePtr',Data);
pLabel= libpointer('doublePtr',Label);
pWeight=libpointer('doublePtr',Weight);
pdW = libpointer('doublePtr',dW);

%prepare data for drawing
idx_0 = find(Label<0.5);
idx_0 = idx_0(1:1000);
idx_1 = find(Label>=0.5);
idx_1 = idx_1(1:1000);

%load so file
loadlibrary liblogistic

figure(1)

t = 0;
for i = 1:200
    tic;
    calllib('liblogistic','parallel_logistic_gradient',...
        pData,pLabel,pWeight,dim,N,pdW,thread_count,batchN,mod(i,batchN));
    t = t + toc();
    
    pWeight.Value = pWeight.Value + alpha.*pdW.Value;
    %Draw |dJdW|
    subplot(2,2,1)
    dJdW(i) = (norm(pdW.Value));
    semilogy(dJdW)
    dps = N*i/t;
    title([num2str(dps/(1e6)) '(MVs)*' num2str(dim) 'pre second'])
    W1(i) = pWeight.Value(1);
    W2(i) = pWeight.Value(2);
    subplot(4,2,5)
    plot(W1)
    title('W(1)')
    subplot(4,2,7)
    plot(W2)
    title('W(2)')
    pause(0.01)
end

unloadlibrary liblogistic

subplot(1,2,2)
hold on
plot(Data(1,idx_0),Data(2,idx_0),'r.')
plot(Data(1,idx_1),Data(2,idx_1),'b.')
x = [-2,8];
y = ((-1)/pWeight.Value(2)).*(1+pWeight.Value(1).*x);
plot(x,y,'g-');
hold off
axis([min(Data(1,:)) max(Data(1,:)) min(Data(2,:)) max(Data(2,:))])
