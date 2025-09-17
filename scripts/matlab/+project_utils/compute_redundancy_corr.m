function [redundancy] = compute_redundancy_corr(features)
% Assumes feature rows= time and feature columns=features
n_features = size(features, 2);
% input validation: features should not contain nans
if any(isnan(features), 'all')
    error("Found NaN values in feature set. Verify the feature extraction output")
end

corr_mat = corr(features);

% While correlation with a flat (i.e. zero-variance) variable is undefined,
% for the purposes of this project, we assume it means no correlation.
corr_mat(isnan(corr_mat)) = 0;

redundancy = sum(abs(corr_mat), 'all') / n_features^2;
end

