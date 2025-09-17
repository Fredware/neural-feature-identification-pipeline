function [relevance] = compute_relevance_corr(features, kinematics_des)
% Assumes rows of features = time and cols = num features
num_features = size(features, 2);
if(any(isnan(features), "all"))
    error("Found NaN values in feature ser. Verify the output of the feature extraction step.")
end

corr_mat = corr(features, kinematics_des);

corr_mat(isnan(corr_mat)) = 0;

% returns relevance for each kinematic
relevance = sum(abs(corr_mat))/num_features;

end