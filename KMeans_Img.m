clear;clc;close all;
loli = imread('loli.jpg');
[x,y,channel] = size(loli);
loli_vec = reshape(loli,x*y,channel);
figure(1)
for i = 1:4
    K=2^i;
    [ ~,cls ] = k_means( double(loli_vec),[],K,[] );
    centers = zeros(K,channel);
    for k = 1:K
        centers(k,:)=mean(loli_vec(find(cls==k),:));
    end
    centers = uint8(centers);
    loli_vec_zip = centers(cls,:);
    loli_zip = reshape(loli_vec_zip,x,y,channel);
    subplot(1,4,i)
    imshow(loli_zip)
end
