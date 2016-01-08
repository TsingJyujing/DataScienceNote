function [ ZCAData,Mu,Lambda,W ] = ZCA( Data,RowVec )
%Data:原始的数据矩阵
%RowVec:如果是行向量形式则是1，否则是0

if RowVec == 1
    Data = Data';
end

[D,N] = size(Data);

if N<=1
    error('数据过少无法进行ZCA')
end

%首先进行中心化
Mu = mean(Data,2);
Lambda = var(Data,0,2);

Data = Data-repmat(Mu(:),1,N);
sigma = Data*Data';
[U,S,~] = svd(sigma);
W = U*diag(1./sqrt(diag(S)+1E-5))*U';
ZCAData = W*Data;

if RowVec == 1
    ZCAData = ZCAData';
end

end

