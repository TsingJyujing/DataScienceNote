clear;clc;close all;
%Generate data
N0 = 1000000;
N1 = 1000000;
x_0 = randn(N0,1)+3;
y_0 = randn(N0,1)+3;
l_0 = zeros(N0,1);
x_1 = randn(N1,1)-3;
y_1 = randn(N1,1)+4;
l_1 = ones(N1,1);
Data = [[x_0,y_0];[x_1,y_1]]';
[dim,N] = size(Data);
Label = [l_0;l_1];
Weight= [1 -1];
dW = zeros(1,2);
save('Datapack.mat')