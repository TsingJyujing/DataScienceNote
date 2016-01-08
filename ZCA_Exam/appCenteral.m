function [ CData ] = appCenteral( Data,RowVec,Mu,Lambda )
%Data:原始的数据矩�?
%RowVec:如果是行向量形式则是1，否则是0
if RowVec == 1
    Data = Data';
end
[~,N] = size(Data);
%首先进行中心�?
Data = Data-repmat(Mu(:),1,N);
CData = Data./repmat(sqrt(Lambda+1E-5),1,N);
if RowVec == 1
    CData = CData';
end
end
