clear;clc;close all;
G1 = randn(100,2).*1+10;
G2 = randn(100,2).*1;
G3 = randn(100,2).*1-10;
X = [G1;G2;G3];
k = 3;
cls = kmeans(X,k);
figure(1)
subplot(1,2,2)
Legends = {'r.','b.','g.'};
hold on
for i = 1:3
    plot(X(find(cls==i),1),X(find(cls==i),2),Legends{i})
end
hold off
title('聚类结果')
subplot(1,2,1)
plot(X(:,1),X(:,2),'.')
title('原始数据')
