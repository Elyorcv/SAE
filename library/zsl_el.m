function [ zsl_accuracy, Y_hit5 ] = zsl_el( S_est, S_te_gt, param)
% ZSL_EL calculates zero-shot classification accuracy
%
% INPUT: 
%    S_est: estimated semantic labels
%    S_te_gt: ground truth semantic labels
%    param: other parameters
%
% Output:  
%    zsl_accuracy: zero-shot classification accuracy (per-sample)

dist     =  1 - (pdist2(S_est, NormalizeFea(S_te_gt')', 'cosine'));
Y_hit5   = zeros(size(dist,1),param.HITK);
for i    = 1:size(dist,1)
    [~, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = param.testclasses_id(I(1:param.HITK));    
end

n = 0;
for i  = 1:size(dist,1)
    if ismember(param.test_labels(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);

end

