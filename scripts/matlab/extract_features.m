function extract_features(varargin)
% --- 1. Parse Name-Value Input Args
p = inputParser;
addParameter(p, 'data_root', '', @isstring);
addParameter(p, 'session_dir', '', @isstring);
addParameter(p, 'training_filename', '', @isstring);
addParameter(p, 'baseline_filename', '', @isstring);
addParameter(p, 'full_stream_filename', '', @isstring);
addParameter(p, 'events_filename', @isstring);
addParameter(p, 'feature_set', '', @isstring);
addParameter(p, 'output_file', '', @isstring);
addParameter(p, 'config_file', '', @isstring);

parse(p, varargin{:})
args = p.Results;

fprintf("Loading configuration files from %s\n", args.config_file);
config_string = fileread(args.config_file);
feature_params = jsondecode(config_string);

session_path = char(fullfile(args.data_root, args.session_dir));

ns5_full_stream_filepath = fullfile(session_path, args.full_stream_filename);
ns2_full_stream_filepath = regexprep(ns5_full_stream_filepath, '.ns5', '.ns2');

rec_start_filepath = fullfile(session_path, sprintf('RecStart_%s.mat', session_path(end-15:end-1)));
fprintf("Calculating NIP offset");
try
    nip_offset_full_stream = project_utils.CalculateNIPOffset_bhm(ns2_full_stream_filepath, rec_start_filepath);
catch e
    fprintf("Failed to calculate NIP offset with RecStart. Attempting SSStruct instead, %s\n", e.message)
    rec_start_filepath = fullfile(session_path, sprintf('Kalman_SSStruct_%s.mat', session_path(end-14:end)));
    nip_offset_full_stream = project_utils.CalculateNIPOffset_bhm(ns2_full_stream_filepath, rec_start_filepath);
    fprintf("SSStruct NIP offset calculation succeeded.");
end

% Load events, kinematic labels, and raw data for the training section of the control task
trial_struct_training = unrl_utils.parseKEF_jag(fullfile(session_path, args.events_filename));
[kdf_kinematics_training, ~, ~, ~, kdf_nip_time_training] = unrl_utils.readKDF_jag(fullfile(session_path, args.training_filename));
nip_range_training = [kdf_nip_time_training(1), kdf_nip_time_training(end)] + nip_offset_full_stream;
fprintf("Loading NS5 Training Data\n");
[ns5_header_training, ns5_data_training] = unrl_utils.fastNSxRead2022('File', ns5_full_stream_filepath, 'Range', nip_range_training);
% Scaling factor for D2A conversion
ns5_scaling_factor_training = (double(ns5_header_training.MaxAnlgVal(1)) - double(ns5_header_training.MinAnlgVal(1))) / ...
    (double(ns5_header_training.MaxDigVal(1)) - double(ns5_header_training.MinDigVal(1)));
NUM_CHANS = 192; % TODO: extract to config.yaml
if size(ns5_data_training, 1) < NUM_CHANS
    NUM_CHANS = size(ns5_data_training, 1); % Adjust for fewer channels if needed
end
ns5_data_training_scaled = single(ns5_data_training(1:NUM_CHANS, :)').*ns5_scaling_factor_training; % Transpose and scale to single precision

% Load raw data for the baseline section of the control task
fprintf("Loading NS5 Basline Data\n");
[~, ~, ~, ~, kdf_nip_time_baseline] = unrl_utils.readKDF_jag(fullfile(session_path, args.baseline_filename));
nip_range_baseline = [kdf_nip_time_baseline(1), kdf_nip_time_baseline(end)] + nip_offset_full_stream;
[ns5_header_baseline, ns5_data_baseline] =  unrl_utils.fastNSxRead2022('File', ns5_full_stream_filepath, 'Range', nip_range_baseline);
ns5_scaling_factor_baseline = (double(ns5_header_baseline.MaxAnlgVal(1)) - double(ns5_header_baseline.MinAnlgVal(1))) / ...
    (double(ns5_header_baseline.MaxDigVal(1)) - double(ns5_header_baseline.MinDigVal(1))); % scale factor for dig2analog
ns5_data_baseline_scaled = single(ns5_data_baseline(1:NUM_CHANS,:)')*ns5_scaling_factor_baseline;


% Extract features
switch(args.feature_set)
    case "NFR"
        features = bhm_nfr.makeNeuralFeatures_NS5( ...
            nfr_spike_threshold, ...
            nip_time_kdf, ...
            nip_time_baseline, ...
            neural_data_ns5, ...
            neural_baseline_ns5 ...
            );
    case "SBP-RAW"
        features = frm_sbp.compute_sbp_features( ...
            nip_time_kdf, ...
            nip_time_baseline, ...
            neural_data_ns5, ...
            neural_baseline_ns5, ...
            mav_kernel_width ...
            )';
    case "DWT-DB1"
    case "DWT-DB4"
        dwt_thresh = frm_wavedec.compute_dwt_thresholds(neural_baseline_ns5);
        features = frm_wavedec.compute_dwt_features( ...
            nip_time_kdf, ...
            neural_data_ns5, ...
            dwt_thresh)';
    case "MAV"
        features = zmh_mav.makeRollingPowerFeatures_zmh( ...
            nip_time_kdf, ...
            nip_time_baseline, ...
            neural_data_ns5, ...
            neural_baseline_ns5, ...
            mav_kernel_width ...
            );
end
save(args.output_file, 'features')
end