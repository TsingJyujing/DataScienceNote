function [ ZCAData ] = appZCA( Data,RowVec,Mu,Lambda,W )
%Data:原始的数据矩阵
%RowVec:如果是行向量形式则是1，否则是0
if RowVec == 1
    Data = Data';
end
[~,N] = size(Data);
%首先进行中心化
Data = Data-repmat(Mu(:),1,N);
%再进行白化
ZCAData = W*Data;
if RowVec == 1
    ZCAData = ZCAData';
end
end