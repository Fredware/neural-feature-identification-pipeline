function [pca_vaf_20, corr_xx, corr_xy, mutual_info_xx, mutual_info_xy] = evaluate_features(trial_features, trial_kinematics)
%EVALUATE_FEATURES Summary of this function goes here
%   Detailed explanation goes here
[~, scores, ~, ~, var_explained, ~] = pca(trial_features); % scores are the projections of X in PC space
pca_vaf_20 = var_explained(1:20)'; % Look at first 20 PCs
corr_xx = project_utils.compute_redundancy_corr(trial_features);
corr_xy = project_utils.compute_relevance_corr(trial_features, trial_kinematics);
mutual_info_xx = 1;
mutual_info_xy = [1:12];
end

