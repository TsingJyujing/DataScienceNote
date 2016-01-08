function [ Data ] = invCenteral( CData,RowVec,Mu,Lambda )
if RowVec == 1
    CData = CData';
end
[~,N] = size(CData);

Data = Data.*repmat(sqrt(Lambda+1E-5),1,N);
Data = Data + repmat(Mu(:),1,N);

if RowVec == 1
    Data = Data';
end
end

