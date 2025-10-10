function extract_features(varargin)
total_timer = tic;
% --- 1. Parse Name-Value Input Args
p = inputParser;
addParameter(p, 'data_root', '', @isstring);
addParameter(p, 'session_dir', '', @isstring);
addParameter(p, 'full_stream_filename', '', @isstring);
addParameter(p, 'baseline_filename', '', @isstring);
addParameter(p, 'kinematics_filepath', '', @isstring);
addParameter(p, 'events_filepath', @isstring);
addParameter(p, 'feature_set_id', '', @isstring);
addParameter(p, 'output_filepath', '', @isstring);
addParameter(p, 'config_filepath', '', @isstring);

parse(p, varargin{:})
args = p.Results;

project_utils.write_log_message('INFO', 'Feature extraction process started', struct('session', args.session_dir, 'feature_set', args.feature_set_id));
project_utils.write_log_message('INFO', 'Loading configuration files');
config_string = fileread(args.config_filepath);
feature_params = jsondecode(config_string);

session_path = char(fullfile(args.data_root, args.session_dir));

project_utils.write_log_message('INFO', 'Computing NIP Offset');
ns5_full_stream_filepath = fullfile(session_path, args.full_stream_filename);
ns2_full_stream_filepath = regexprep(ns5_full_stream_filepath, '.ns5', '.ns2');
rec_start_filepath = fullfile(session_path, sprintf('RecStart_%s.mat', session_path(end-14:end)));
try
    nip_offset_full_stream = project_utils.CalculateNIPOffset_bhm(ns2_full_stream_filepath, rec_start_filepath);
catch e
    project_utils.write_log_message('WARN', sprintf("Failed to compute NIP offset with RecStart. Attempting SSStruct instead. %s", e.message));
    rec_start_filepath = fullfile(session_path, sprintf('Kalman_SSStruct_%s.mat', session_path(end-14:end)));
    nip_offset_full_stream = project_utils.CalculateNIPOffset_bhm(ns2_full_stream_filepath, rec_start_filepath);
    project_utils.write_log_message('INFO', 'SSStruct computation succeeded');
end

% Load events, kinematic labels, and raw data for the training section of the control task
% trial_struct_training = unrl_utils.parseKEF_jag(fullfile(session_path, args.events_filename));
% [kdf_kinematics_training, ~, ~, ~, kdf_nip_time_training] = unrl_utils.readKDF_jag(fullfile(session_path, args.training_filename));
kdf_nip_time_training = h5read(args.kinematics_filepath, "/nip_time");
nip_range_training = [kdf_nip_time_training(1), kdf_nip_time_training(end)] + nip_offset_full_stream;
project_utils.write_log_message('INFO', 'Loading NS5 training data...')
load_timer = tic;
[ns5_header_training, ns5_data_training] = unrl_utils.fastNSxRead2022('File', ns5_full_stream_filepath, 'Range', nip_range_training);
project_utils.write_log_message('INFO', 'Training data loaded', struct('duration_sec', toc(load_timer)))
s_raw = whos('ns5_data_training');
project_utils.write_log_message('DEBUG', 'Memory usage for raw training data.', struct('variable', 'ns5_data_training', 'megabytes', s_raw.bytes / 1024^2));
% Scaling factor for D2A conversion
ns5_scaling_factor_training = (double(ns5_header_training.MaxAnlgVal(1)) - double(ns5_header_training.MinAnlgVal(1))) / ...
    (double(ns5_header_training.MaxDigVal(1)) - double(ns5_header_training.MinDigVal(1)));
NUM_CHANS = 192; % TODO: extract to config.yaml
if size(ns5_data_training, 1) < NUM_CHANS
    NUM_CHANS = size(ns5_data_training, 1); % Adjust for fewer channels if needed
end
ns5_data_training_scaled = single(ns5_data_training(1:NUM_CHANS, :)').*ns5_scaling_factor_training; % Transpose and scale to single precision
s = whos('ns5_data_training_scaled');
project_utils.write_log_message('DEBUG', 'Memory usage for scaled training data', struct('variable', 'ns5_data_training_scaled', 'megabytes', s.bytes / 1024^2));

% Load raw data for the baseline section of the control task
[~, ~, ~, ~, kdf_nip_time_baseline] = unrl_utils.readKDF_jag(fullfile(session_path, args.baseline_filename));
nip_range_baseline = [kdf_nip_time_baseline(1), kdf_nip_time_baseline(end)] + nip_offset_full_stream;
project_utils.write_log_message('INFO', 'Loading NS5 baseline data...');
load_timer = tic;
[ns5_header_baseline, ns5_data_baseline] =  unrl_utils.fastNSxRead2022('File', ns5_full_stream_filepath, 'Range', nip_range_baseline);
project_utils.write_log_message('INFO', 'Baseline data loaded', struct('duration_sec', toc(load_timer)));
s_raw = whos('ns5_data_baseline');
project_utils.write_log_message('DEBUG', 'Memory usage for raw training data.', struct('variable', 'ns5_data_baseline', 'megabytes', s_raw.bytes / 1024^2));
ns5_scaling_factor_baseline = (double(ns5_header_baseline.MaxAnlgVal(1)) - double(ns5_header_baseline.MinAnlgVal(1))) / ...
    (double(ns5_header_baseline.MaxDigVal(1)) - double(ns5_header_baseline.MinDigVal(1))); % scale factor for dig2analog
ns5_data_baseline_scaled = single(ns5_data_baseline(1:NUM_CHANS,:)')*ns5_scaling_factor_baseline;
s = whos("ns5_data_baseline_scaled");
project_utils.write_log_message('DEBUG', 'Memory usage for scaled baseline data', struct('variable', 'ns5_data_baseline_scaled', 'megabytes', s.bytes / 1024^2));

% Extract features
project_utils.write_log_message('INFO', 'Commencing feature extraction', struct('feature_set', args.feature_set_id));
feature_timer = tic;
switch(args.feature_set_id)
    case "NFR"
        % returns 192xN, and 1xN. TODO: Transpose results to follow the
        % row=time convention
        [features, ~, frame_computation_times] = bhm_nfr.makeNeuralFeatures_NS5( ...
            feature_params.nfr.spike_threshold_std, ...
            kdf_nip_time_training, ...
            kdf_nip_time_baseline, ...
            ns5_data_training_scaled, ...
            ns5_data_baseline_scaled ...
            );
        features = features'; frame_computation_times = frame_computation_times';
    case "SBP-RAW"
        % returns Nx192, and 1XN. TODO: transpose times to follow the
        % row=time convention
        [features, frame_computation_times] = frm_sbp.compute_sbp_features( ...
            kdf_nip_time_training, ...
            kdf_nip_time_baseline, ...
            ns5_data_training_scaled, ...
            ns5_data_baseline_scaled, ...
            feature_params.sbp ...
            );
        frame_computation_times = frame_computation_times';
    case "DWT-DB1"
    case "DWT-DB4"
        dwt_thresh = frm_wavedec.compute_dwt_thresholds(ns5_data_baseline_scaled);
        % returns Nx1920 and 1xN. TODO: transpose times to follow the
        % times=rows convention
        [features, frame_computation_times] = frm_wavedec.compute_dwt_features( ...
            kdf_nip_time_training, ...
            ns5_data_training_scaled, ...
            dwt_thresh);
        frame_computation_times = frame_computation_times';
    case "MAV"
        % returns 192xN, and 1xN. TODO: Transpose results to follow
        % row=time convention
        [features, frame_computation_times] = zmh_mav.makeRollingPowerFeatures_zmh( ...
            kdf_nip_time_training, ...
            kdf_nip_time_baseline, ...
            ns5_data_training_scaled, ...
            ns5_data_baseline_scaled, ...
            feature_params.mav ...
            );
        features = features'; frame_computation_times = frame_computation_times';
end
extraction_duration = toc(feature_timer);
s = whos('features');
project_utils.write_log_message('INFO', 'Feature extraction completed', struct('duration_sec', extraction_duration, 'megabytes', s.bytes/1024^2, 'num_features', size(features, 2)));

project_utils.write_log_message('INFO', 'Writing features to HDF5 file', struct('path', args.output_filepath));
save_timer = tic;
[output_dir, ~, ~] = fileparts(args.output_filepath);
if ~exist(output_dir, "dir")
    mkdir(output_dir)
end
h5create(args.output_filepath, '/features', size(features));
h5write(args.output_filepath, '/features', features);
h5create(args.output_filepath, '/computation_times', size(frame_computation_times));
h5write(args.output_filepath, '/computation_times', frame_computation_times);
if exist(args.output_filepath, 'file')
    project_utils.write_log_message('INFO', 'HDF5 file successfully written.', struct('duration_sec', toc(save_timer)));
else
    project_utils.write_log_message('ERROR', 'Failed to write HDF5 file.', struct('path', args.output_filepath));
end

project_utils.write_log_message('INFO', 'Feature extraction process finished.', struct('total_duration_sec', toc(total_timer)));
end