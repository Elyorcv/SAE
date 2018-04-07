%% %%% CUB DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Following code shows a demo for aPY dataset to reproduce the result of the paper:
% 
% Semantic Autoencoder for Zero-shot Learning. 
% 
% Elyor Kodirov, Tao Xiang, and Shaogang Gong, CVPR 2017.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We used GoogleNet features.
clc
clc, clear all,  close all

addpath('library')
load('cub_demo_data.mat');

% Dimension reduction
Y    = label_matrix(train_labels_cub')';
W    = (X_tr'*X_tr + 50 * eye(size(X_tr'*X_tr)))^(-1)*X_tr'*Y;
X_tr = X_tr * W;
X_te = X_te * W;

% Learn projection
lambda  = .2;
S_tr    = NormalizeFea(S_tr);
W       = SAE(X_tr', S_tr', lambda)';



%%%%% Testing %%%%%
%[F --> S], projecting data from feature space to semantic space 
S_te_est = X_te * W;
dist     =  1 - (pdist2((S_te_est), (S_te_pro), 'cosine'));
dist     = zscore(dist);
HITK     = 1;
Y_hit5   = zeros(size(dist,1),HITK);
for i    = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(test_labels_cub(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('\n[1] aPY ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);


%[S --> F], projecting from semantic to visual space
dist    =  1 - zscore(pdist2(X_te, (S_te_pro * W'), 'cosine')) ;
dist    = zscore(dist);
HITK    = 1;
Y_hit5  = zeros(size(dist,1),HITK);
for i   = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(test_labels_cub(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('[2] aPY ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);








