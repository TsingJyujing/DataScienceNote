function [ Data ] = invZCA( ZCAData,RowVec,Mu,Lambda,W )
if RowVec == 1
    ZCAData = ZCAData';
end
[~,N] = size(ZCAData);

Data = W\ZCAData;
Data = Data + repmat(Mu(:),1,N);

if RowVec == 1
    Data = Data';
end
end

