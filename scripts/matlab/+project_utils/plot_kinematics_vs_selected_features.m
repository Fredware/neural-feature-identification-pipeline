function [fig_h] = plot_kinematics_vs_selected_features(kinematics, features, feature_idxs, feature_key, features_selected_idxs)

features_selected = features(:, feature_idxs{feature_key});
features_selected = features_selected(:, features_selected_idxs);
KINEMATIC_OFFSETS = [1:12]; % DC offset for visualization purposes
FEATURE_OFFSETS = [1:size(features_selected, 2)];

fig_h = figure();
ax_list = [];

tlo_h = tiledlayout(3,1);
ax_h = nexttile;
plot(kinematics + KINEMATIC_OFFSETS);
ax_list = [ax_list, ax_h];
title(ax_h, 'Desired Kinematics')

ax_h = nexttile;
imagesc(normalize(features_selected, 'range')')
colormap('sky')
ax_list = [ax_list, ax_h];
title(ax_h, 'Normalized Features')

ax_h = nexttile;
plot(features_selected + FEATURE_OFFSETS);
ax_list = [ax_list, ax_h];
linkaxes(ax_list, 'x')
title(ax_h, 'Unscaled Features')

title(tlo_h, "Kinematics Vs. Selected Features")
subtitle(tlo_h, feature_key)

end

