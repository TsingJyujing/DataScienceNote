function [ CData,Mu,Lambda ] = Centeral( Data,RowVec )
%Data:原始的数据矩�?
%RowVec:如果是行向量形式则是1，否则是0

if RowVec == 1
    Data = Data';
end

[D,N] = size(Data);

if N<=1
    error('数据过少无法进行Centeral')
end

%首先进行中心�?
Mu = mean(Data,2);
Lambda = var(Data,0,2);
Data = Data-repmat(Mu(:),1,N);
CData = Data./repmat(sqrt(Lambda+1E-5),1,N);

if RowVec == 1
    CData = CData';
end

end

