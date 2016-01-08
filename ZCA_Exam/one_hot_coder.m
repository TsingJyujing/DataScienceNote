function one_hot = one_hot_coder(label)
%MNIST数据集专用的One-Hot编码器
one_hot = zeros(10,length(label));
fixpos = 1:length(label);
fixpos = fixpos - 1;
fixpos = fixpos * 10;
one_hot(fixpos+label+1)=1;
end