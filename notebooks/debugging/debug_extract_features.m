function debug_extract_features(job_id, feature_set_id)
clear vars; clc; close all;

[current_dir, ~, ~] = fileparts(mfilename("fullpath"));
project_root = fullfile(current_dir, '..', '..');

config_json_filepath = string(fullfile(project_root, 'workflow', 'matlab_config.json'));
manifest_filepath = fullfile(project_root, 'reports', 'manifest.tsv');
matlab_scripts_path = fullfile(project_root, 'scripts', 'matlab');

data_root = "/uufs/chpc.utah.edu/common/home/george-group1/data-usea";
scratch_root = "/scratch/general/vast/u1424875/neural-feature-identification-pipeline";

opts = detectImportOptions(manifest_filepath, 'FileType','text');
opts = setvartype(opts, 'job_id', 'double');
opts = setvartype(opts, {'participant_id', 'session_dir', 'training_filename', 'baseline_filename', 'events_filename', 'full_stream_filename'}, 'string');
manifest = readtable(manifest_filepath, opts);

job_info = manifest(manifest.job_id == job_id, :);
if isempty(job_info)
    error("JOB ID %d not found in %s", job_id, manifest_filepath)
end

kinematics_filepath = fullfile(scratch_root, num2str(job_id), "kinematics_debug.h5");
events_filepath = fullfile(scratch_root, num2str(job_id), "events_debug.h5");
output_dir = fullfile(scratch_root, num2str(job_id), "features");
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
output_features_filepath = fullfile(output_dir, sprintf('%s_debug.h5', feature_set_id));

addpath(matlab_scripts_path);

extract_features( ...
    'data_root', data_root, ...
    'session_dir', job_info.session_dir, ...
    'full_stream_filename', job_info.full_stream_filename, ...
    'baseline_filename', job_info.baseline_filename, ...
    'kinematics_filepath', kinematics_filepath, ...
    'events_filepath', events_filepath, ...
    'feature_set_id', feature_set_id, ...
    'output_filepath', output_features_filepath, ...
    'config_filepath', config_json_filepath ...
    );
end