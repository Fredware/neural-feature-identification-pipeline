function [outputArg1,outputArg2] = plot_manifest_kinematics(manifest_file, job_id)
%PLOT_MANIFEST_KINEMATICS iterates over the datasets in manifest_file, and
%plots the kinematics of the TrainingData.
config_string = fileread("../../config.json");
config_json = jsondecode(config_string);
data_root = config_json.paths.data_root;
manifest_tbl = readtable(manifest_file, 'FileType','text', 'Delimiter', '\t');

set(groot, 'Units', 'pixels');
screenSize = get(groot, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

fig_width = (20/21)*screenWidth;
fig_height =  0.9*screenHeight;
fig_left = screenWidth - fig_width;
fig_bottom = 1;
figurePosition = [fig_left, fig_bottom, fig_width, fig_height];

for i = job_id:numel(manifest_tbl)
    kdf_filepath =string(fullfile(data_root,manifest_tbl.session_dir(i), manifest_tbl.training_filename(i)));
    [kinematics,~,~,~,~] = unrl_utils.readKDF_jag(kdf_filepath);
    VIS_OFFSET = [1:12];
    fig_h = figure();
    set(fig_h, 'Units', 'pixels', 'Position', figurePosition, 'Visible', 'on');
    plot(0.5*kinematics'+VIS_OFFSET, LineWidth=1, Marker=".");
    title(string(manifest_tbl.participant_id(i)) + string(manifest_tbl.training_filename(i)), 'Interpreter','none')
    subtitle(string(manifest_tbl.job_id(i)))
    legend();

    fprintf('Press Enter to continue to the next image...\n');
    pause;
    close(fig_h);
end