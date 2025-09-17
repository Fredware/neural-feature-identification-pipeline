function [fig_h] = plot_kinematics_predicted_vs_selected_features(kinematics, test_mask, test_trial_idxs, kinematics_predicted, features, feature_key, feature_idxs, feature_selected_idxs)
KINEMATIC_OFFSETS = [1:12]; % DC offset for visualization purposes

features_selected = features(:, feature_idxs{feature_key});
features_selected = features_selected(:, feature_selected_idxs);

fig_h = figure();
ax_list = [];

tlo_h = tiledlayout(4,1);
ax_h = nexttile;
hold on
plot(kinematics(test_mask, :) + KINEMATIC_OFFSETS);
xline(test_trial_idxs.relative_starts, 'Color','red');
xline(test_trial_idxs.relative_stops);
hold off
ax_list = [ax_list, ax_h];
title(ax_h, "Test Kinematics Desired");

ax_h = nexttile;
plot(kinematics_predicted + KINEMATIC_OFFSETS);
ax_list = [ax_list, ax_h];
title(ax_h, "Test Kinematics Predicted");
linkaxes(ax_list, 'xy')

ax_h = nexttile;
imagesc(normalize(features_selected(test_mask, :),'range')');
colormap('sky')
ax_list = [ax_list, ax_h];
title(ax_h, "Test Features Normalized")

ax_h = nexttile;
plot(features_selected(test_mask,:))
ax_list = [ax_list, ax_h];
title(ax_h, "Test Features Unscaled")
linkaxes(ax_list, 'x')

title(tlo_h, "Intended Vs. Predicted Kinematics")
subtitle(tlo_h, feature_key)

end

