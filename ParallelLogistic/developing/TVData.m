clear
clc
close all
N =1000;
figure(1)
hold on
x = randn(N,1)*3+3;
y = randn(N,1)*3-3-x;
plot(x,y,'b.')
x = randn(N,1)*3-8;
y = randn(N,1)*2-8+0.3*x;
plot(x,y,'r.')
hold off