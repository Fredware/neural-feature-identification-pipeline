function plot_kinematics_vs_features(kinematics, features, feature_idxs)
cmap_opts = "sky";
features_plot = normalize(features, 1, "range"); % Normalize each column. Assumes features is NxChans
t = 1:5:length(features);
fig_h = figure();
tlo_h = tiledlayout(5,1);
ax1_h = nexttile; plot(kinematics(t, :)+ [1:12]); set(ax1_h, "YDir", "reverse");
ax2_h = nexttile; imagesc(features_plot(t, feature_idxs{"NFR"})'); title("NFR");
ax3_h = nexttile; imagesc(features_plot(t, feature_idxs{"MAV"})'); title("MAV");
ax4_h = nexttile; imagesc(features_plot(t, feature_idxs{"SBP"})'); title("SBP");
ax5_h = nexttile; imagesc(features_plot(t, feature_idxs{"DWT"})'); title("DWT");
colormap(cmap_opts{:});
linkaxes([ax1_h, ax2_h, ax3_h, ax4_h, ax5_h], 'x')

plot_opts = { "LineStyle", ":"};
fig_h = figure();
tlo_h = tiledlayout(5,1);
ax1_h = nexttile; plot(kinematics(t,:) + [1:12]); %set(ax1_h, "YDir", "reverse");
ax2_h = nexttile; plot(features_plot(t, feature_idxs{"NFR"}), plot_opts{:}); title("NFR");
ax3_h = nexttile; plot(features_plot(t, feature_idxs{"MAV"}), plot_opts{:}); title("MAV");
ax4_h = nexttile; plot(features_plot(t, feature_idxs{"SBP"}), plot_opts{:}); title("SBP");
ax5_h = nexttile; plot(features_plot(t, feature_idxs{"DWT"}), plot_opts{:}); title("DWT");
linkaxes([ax1_h, ax2_h, ax3_h, ax4_h, ax5_h], 'x')
end