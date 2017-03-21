%% %%% Imagenet-2 DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Following code shows a demo for Imagenet-2 dataset to reproduce the result of the paper:
% 
% Semantic Autoencoder for Zero-shot Learning. 
% 
% Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We used GoogleNet features.
% X_tr: train data Ntr by d matrix. Ntr is the number of train samples, d
% is the feature dimension size.
% X_te: test  data Nte by d matrix. Nte is the number of test samples. 
% Y: Ntr by c label matrix (consists of one-hot vector). c is the number of classes. 
% S_tr is Ntr by k matrix. k is the semantic dimension size. 

clc
clear all
warning off

% Loading the data
addpath('library')
load('ImNet_2_demo_data.mat')

%% Dimension reduction
W    = (X_tr'  * X_tr + 150*eye(size(X_tr'*X_tr)))^(-1)*X_tr'*(Y)  ;
X_tr = X_tr * W;
X_te = X_te * W;

%% Learn projection
lambda   = 5;
W        = SAE(X_tr', S_tr', lambda)';

%%%%% Testing, nearest neighbour classification %%%%%
%[F --> S], projecting data from feature space to semantic space 
S_te_est = X_te * W;
dist     =  1 -  pdist2(zscore(S_te_est),zscore(S_te_pro')', 'cosine');  % 26.3%
HITK     = 5;
Y_hit5   = zeros(size(dist,1),HITK);
for i  = 1:size(dist,1)
    [~, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(Y_te(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('\n[1] ImageNet-2 ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);


%[S --> F], projecting from semantic to visual space
dist  =  1 - (pdist2(X_te,zscore(W * S_te_pro')', 'cosine')) ;    % 27.1%
HITK   = 5;
Y_hit5 = zeros(size(dist,1),HITK);
for i  = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = X_te_cl_id(I(1:HITK));
end
n = 0;
for i  = 1:size(dist,1)
    if ismember(Y_te(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('[2] ImageNet-2 ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);

clear i n Y_hit5 zsl_accuracy sort_dist_i lambda HITK dist I