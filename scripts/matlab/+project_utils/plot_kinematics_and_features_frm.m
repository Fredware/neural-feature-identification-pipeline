function [fig_h] = plot_kinematics_and_features_frm(kdf_filename, mask_name, kinematics_train, kinematics_test, kinematic_sets_predicted, feature_sets_names, feature_sets_train, feature_sets_test, feature_sets_idxs)
%PLOT_KINEMATICS_AND_FEATURES Summary of this function goes here
%   Detailed explanation goes here
num_feature_sets = numel(feature_sets_names);
fig_name = regexprep(kdf_filename, '\.kdf$', mask_name);

fig_h = figure('Visible','off');
tlo = tiledlayout(num_feature_sets+1, 2); % First row for kinematics, 1 col for training and another for testing
title(tlo, fig_name);

plot_kinematics(tlo, kinematics_train, kinematics_test, kinematic_sets_predicted, feature_sets_names);
plot_features(tlo, feature_sets_train, feature_sets_test, feature_sets_idxs, feature_sets_names);
end

function plot_kinematics(tlo, kinematics_train, kinematics_test, kinematics_sets_predicted, feature_sets_names)
ax = nexttile(tlo);
plot(kinematics_train', "DisplayName","Train Kinematics", "LineWidth",3);
xlim([0, length(kinematics_train)+1]);
legend(ax)
ylim(1.1*[min(kinematics_train, [], "all"), max(kinematics_train, [], "all")]);

ax = nexttile(tlo);
hold(ax, 'on')
plot(kinematics_test', "DisplayName", "Test Kinematics", "LineWidth", 3);
ylim(1.1*[min(kinematics_test, [], "all"), max(kinematics_test, [], "all")]);

for ii = 1:numel(feature_sets_names)
    plot(kinematics_sets_predicted{ii}', "DisplayName", feature_sets_names{ii}, 'LineWidth', 2);
end
hold(ax, "off");
legend(ax);
end

function plot_features(tlo, feature_sets_train, feature_sets_test, feature_sets_idxs, feature_sets_names)
for ii = 1:numel(feature_sets_names)
    if isempty(feature_sets_idxs{ii})
        features_sel_plt = 1:50;
    else
        num_sel = numel(feature_sets_idxs{ii});
        features_sel_plt = feature_sets_idxs{ii}(1:min(10,num_sel));
    end

    nexttile(tlo);
    features_plt_train = feature_sets_train{ii}(features_sel_plt, :);
    [p_lo_train, p_hi_train] = compute_percentiles(features_plt_train);
    imagesc(features_plt_train);
    clim([p_lo_train, p_hi_train]);
    colormap("sky");
    ylabel(feature_sets_names{ii})

    nexttile(tlo);
    features_plt_test = feature_sets_test{ii}';
    features_plt_test = features_plt_test(:, features_sel_plt);
    [p_lo_test, p_hi_test] = compute_percentiles(features_plt_test);
    plot(features_plt_test, 'Color', [0, 0.45, 0.75, 0.25], 'LineWidth', 2);
    if (p_lo_test ~= p_hi_test)
        ylim([p_lo_test, p_hi_test]);
    end
end
end

function [p_lo, p_hi] = compute_percentiles(features)
p_hi = prctile(max(features, [], 2), 95);
p_lo = prctile(min(features, [], 2), 05);
end
