function [ k_means_result,cls_vec ] = k_means( data_set,weight,K,plot2pos )
%% 说明的言叶
%{
    data_set:输入的数据集，以行向量为单位
    weight:输入的权重，如果是空就自动设为各个数据权值相等
    K:K-Means的K还需要我明说么~，~
    plot2pos:用来绘图的向量，大小N*2的矩阵，如果是空就不绘图了。(要不然超过三维怎么绘图)
%}

%% 初始化各项参数
[N,dim] = size(data_set);
if isempty(weight)
    weight = ones(N,1);
else
    weight = weight(:);
end
if isempty(plot2pos)
    isdisplay = 0;
else
    isdisplay = 1;
end

% 初始化点集
group_center = mean(data_set);
group_range = range(data_set);
centers = (randn(K,dim).*repmat(group_range,K,1)./3+repmat(group_center,K,1));

%各个变量的初始化
dist_vec = zeros(N,K);
rep_tms = 0;
delay = 1e-3;
dumper = 0.1;
p2pc = zeros(K,2);
%J = 0;
logJ = [];
minJ = -1;

%% 如果画图，为画图设置各项Legend（颜色）
if isdisplay == 1
    Style = {'o'};
    Color = {'r','g','b','m','k','c','y'};
    Legends = {};
    for i = 1:length(Style)
        for j = 1:length(Color)
            Legends = [Legends;[Style{i},Color{j}]];
        end
    end
end

%% 迭代过程
while(1)
    
    for i = 1:K
        %求到每一个中心的距离向量，且不用张量求更加节约空间
        dist_vec(:,i) = sqrt(sum(...
            (data_set - repmat(centers(i,:),N,1)).^2,2)...
            );
    end
    %求出每一个点离哪一个中心更近。
    [~,cls_vec] = min(dist_vec,[],2);
    lst_cnt = centers;
    for i = 1:K
        %根据权重重新计算中心
        cls_idx = find(cls_vec==i);
        if isempty(cls_idx)
            centers(i,:) = (randn(1,dim).*group_range./3+group_center);
        else
            centers(i,:) = sum(data_set(cls_idx,:).*...
                repmat(weight(cls_idx),1,dim))...
                ./sum(weight(cls_idx));
        end
    end
    
    % 计算代价函数
    CMat = (data_set-lst_cnt(cls_vec,:)).^2;
    J = sum(CMat(:));
    
    % 根据阻尼比更新
    centers = (1-dumper).*centers + dumper.*lst_cnt;
    
    if minJ<0 || J<minJ
        minJ=J;
    end
    
    if J>=minJ
        rep_tms = rep_tms+1;
    else
        rep_tms = 0;
    end
    
    if rep_tms>=5
        break;
    end
    
    if isdisplay == 1
        % 绘制代价函数及其差分
        figure(2)
        logJ = [logJ J]; %#ok<AGROW>
        subplot(2,1,1)
        plot(logJ)
        axis tight
        subplot(2,1,2)
        plot(diff(logJ))
        
        % 绘制分布图
        figure(1)
        cla
        hold on
        for i = 1:K
            cls_idx = find(cls_vec==i);
            if isempty(cls_idx)
                p2pc(i,:) = mean(plot2pos);
            else
                p2pc(i,:) = sum(...
                        plot2pos(cls_idx,:).*...
                        repmat(weight(cls_idx),1,2)...
                    )...
                    ./sum(weight(cls_idx));
            end
            plot(p2pc(i,1),p2pc(i,2),Legends{1+mod(i,length(Legends))});
            scatter(plot2pos(cls_idx,1),plot2pos(cls_idx,2),...坐标
                weight(cls_idx)./max(weight).*10,...大小
                Color{1+mod(i,length(Color))},'filled');%颜色
        end
        
        hold off
        pause(delay)
    end
end

%% 输出数据集
not_empty_cls = unique(cls_vec);
k_means_result = cell(length(not_empty_cls),1);
for i = 1:length(not_empty_cls)
    cls = not_empty_cls(i);
    sub_res = cell(3,1);
    cls_idx = find(cls_vec==cls);
    sub_res{1} = centers(cls,:);
    sub_res{3} = cls_idx;
    sub_res{2} = data_set(cls_idx,:);
    k_means_result{i} = sub_res;
end

end

