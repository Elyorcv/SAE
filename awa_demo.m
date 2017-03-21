%% %%% AwA DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Following code shows a demo for AwA dataset to reproduce the result of the paper:
% 
% Semantic Autoencoder for Zero-shot Learning. 
% 
% Elyor Kodirov, Tao Xiang, and Shaogang Gong
% To appear in CVPR 2017.
% 
%
% You are supposed to get following:
% [1] AwA ZSL accuracy [V >>> S]: 84.7%
% [2] AwA ZSL accuracy [S >>> V]: 84.0%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% We used GoogleNet features.
clc, clear all,  close all

addpath('data_zsl');
addpath('library');

%%%%% Load the data
load('data_zsl/awa_demo_data.mat');
X_tr    = NormalizeFea(X_tr')';


%%%%% Training
% SAE
lambda  = 500000;
W       = SAE(X_tr', S_tr', lambda);


%%%%% Test %%%%% 
param.HITK           = 1;
param.testclasses_id = param.testclasses_id;
param.test_labels    = param.test_labels;

%[F --> S], projecting data from feature space to semantic sapce 
S_est        = X_te * NormalizeFea(W)'; 
[zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param); 

fprintf('\n[1] AwA ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);

%[S --> F], projecting from semantic to visual space
X_te_pro     = NormalizeFea( S_te_pro')' * NormalizeFea(W);
[zsl_accuracy]= zsl_el(X_te, X_te_pro, param); 
fprintf('[2] AwA ZSL accuracy [S >>> V]: %.1f%%\n', zsl_accuracy*100);










