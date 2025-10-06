%==============================================================================
% FUNCTION: compute_sbp_features
%
% PURPOSE:
%   Computes spiking-band power features from NS5 neural data, subtracting a baseline,
%   simulating a 30 Hz processing loop for neural decoding applications. The implementation
%   follows the methods described in Nason et al. 2020 (Nature BME)
%
% INPUTS:
%   kdf_nip_time        - [double vector] Timestamps for main neural data (s).
%   baseline_nip_time   - [double vector] Timestamps for baseline data (s).
%   neural_data         - [double matrix] Neural data (time x channels).
%   baseline_neural_data- [double matrix/string] Baseline data or 'None'.
%   kernel_width        - [double scalar] Window size for feature computation (s).
%
% OUTPUTS:
%   neural_feature      - [double matrix] Computed neural features (channels x time).
%
% ASSUMPTIONS AND LIMITATIONS:
%   - Neural data is sampled at 30 kHz.
%   - Channels are typically 192.
%   - A 30 Hz loop (33 ms) is simulated.
%   - Filter parameters are loaded from predefined files.
%
% REFERENCES:
%   - Based on makeKDF_NS5_PTZ file, simplified for NS5 feature recreation.
%
% MODIFICATION HISTORY:
%   2018-09-18  MRB  Created original version for MAV features.
%   2025-04-11  Grok Refactored for modularity, NASA conventions, fixed loop time bug.
%   2025-04-12  FRM  Modified the code to compute SBP instead of MAV
%==============================================================================
function [neural_feature] = compute_sbp_features(kdf_nip_timestamps, ...
    baseline_nip_time, neural_data, baseline_neural_data, feature_params)

% Load filter parameters
% load('../src/filter_params/lfpSOSfilter20180417.mat');
% load('../src/filter_params/lfpSOSnotchfilter20180417.mat');

% Constants
LOOP_TIME_SECONDS = 33e-3; % [sec] (33 [ms]) simulated software loop time
MAX_DATA_LOOKBACK = 37 * 30; % Maximum samples to look back
DOWNSAMPLE_30KHZ_TO_2KHZ = 15; % 30e3 / 2e3
DOWNSAMPLE_2KHZ_TO_30HZ = 66; % 2e3 / (1/33e-3)
FRAME_LEN = 50e-3 * 2e3; % [sec] * (FS: 2e3)

neural_data_filt = apply_bandpass_filter(neural_data);
n_chans = size(neural_data_filt, 2);
% neural_nip_timestamps = linspace(kdf_nip_timestamps(1), kdf_nip_timestamps(end), length(neural_data_filt));
neural_data_2k = downsample(neural_data_filt, DOWNSAMPLE_30KHZ_TO_2KHZ);
neural_data_2k_idxs = 1:length(neural_data_2k);
neural_data_30h_idxs = downsample(neural_data_2k_idxs, DOWNSAMPLE_2KHZ_TO_30HZ);

frame_idxs = find(ismember(neural_data_2k_idxs, neural_data_30h_idxs));
n_frames = length(frame_idxs);
neural_feature = nan(n_frames, n_chans);
for i=1:n_frames
    frame_idx = frame_idxs(i);
    if frame_idx < FRAME_LEN
        temp = neural_data_2k(1:frame_idx, :);
        frame_data = [zeros(FRAME_LEN - size(temp, 1), n_chans); temp];
    else
        frame_data = neural_data_2k(frame_idx-FRAME_LEN+1: frame_idx, :);
    end
    neural_feature(i, :) = mean(abs(frame_data));
end

n_frames_kdf = length(kdf_nip_timestamps);
if n_frames < n_frames_kdf
    neural_feature = padarray(neural_feature, n_frames_kdf-n_frames, 'replicate', 'post');
elseif n_frames > n_frames_kdf
    neural_feature = neural_feature(1:n_frames_kdf,:);
end

% % Compute baseline
% neural_baseline = compute_neural_baseline(baseline_neural_data, ...
%     baseline_nip_time, kernel_width, LOOP_TIME_SECONDS, ...
%     MAX_DATA_LOOKBACK);
% 
% % Process neural features
% neural_feature = process_neural_features(neural_data, kdf_nip_timestamps, ...
%     kernel_width, LOOP_TIME_SECONDS, MAX_DATA_LOOKBACK, ...
%     neural_baseline);

disp('Neural feature processing completed.')
end

%==============================================================================
% FUNCTION: compute_neural_baseline
%
% PURPOSE:
%   Computes baseline neural features by filtering and averaging neural data
%   over a sliding window, simulating a 30 Hz loop.
%
% INPUTS:
%   baseline_neural_data - [double matrix/string] Baseline data or 'None'.
%   baseline_nip_time    - [double vector] Timestamps for baseline data (s).
%   kernel_width         - [double scalar] Window size for feature computation (s).
%   loop_time_seconds    - [double scalar] Simulated loop time (s).
%   max_data_lookback    - [integer scalar] Maximum samples to look back.
%
% OUTPUTS:
%   neural_baseline      - [double vector] Baseline feature per channel.
%
% ASSUMPTIONS AND LIMITATIONS:
%   - Assumes 30 kHz sampling rate.
%   - Uses a 4th-order Butterworth high-pass filter at 750 Hz.
%
% MODIFICATION HISTORY:
%   2025-04-11  Grok  Extracted from make_rolling_power_features_zmh.
%
%==============================================================================
function neural_baseline = compute_neural_baseline(baseline_neural_data, ...
    baseline_nip_time, kernel_width, loop_time_seconds, max_data_lookback)

if strcmp(baseline_neural_data, 'None')
    neural_baseline = 0; % No baseline
    return;
end

% Filter baseline data
filtered_baseline_data = apply_bandpass_filter(baseline_neural_data);

% Simulate 30 Hz loop
baseline_indices_30k = ceil(baseline_nip_time - baseline_nip_time(1)) + 1;
baseline_index_diffs = zeros(size(baseline_indices_30k));
baseline_index_diffs(1:end-1) = diff(baseline_indices_30k);
baseline_index_diffs(baseline_index_diffs > max_data_lookback) = max_data_lookback;

% Initialize buffer
baseline_neural_buffer = zeros(size(filtered_baseline_data, 2), ...
    floor(kernel_width / loop_time_seconds));

% Compute baseline features
baseline_neural_kdf = zeros(size(filtered_baseline_data, 2), ...
    length(baseline_indices_30k)-1);
for time_idx = 2:length(baseline_indices_30k)
    start_timestamp = baseline_indices_30k(time_idx) - baseline_index_diffs(time_idx-1);
    end_timestamp = baseline_indices_30k(time_idx) - 1;
    baseline_neural_subset = filtered_baseline_data(start_timestamp:end_timestamp, :);
    baseline_neural_buffer = circshift(baseline_neural_buffer, ...
        [0, length(baseline_neural_subset)]);
    baseline_neural_buffer(:, 1:length(baseline_neural_subset)) = baseline_neural_subset';
    baseline_neural_kdf(:, time_idx-1) = mean(abs(baseline_neural_buffer), 2);
end

neural_baseline = mean(baseline_neural_kdf, 2);
end

%==============================================================================
% FUNCTION: process_neural_features
%
% PURPOSE:
%   Processes neural data to compute rolling power features, subtracting a
%   baseline, simulating a 30 Hz loop.
%
% INPUTS:
%   neural_data         - [double matrix] Neural data (time x channels).
%   kdf_nip_time        - [double vector] Timestamps for neural data (s).
%   kernel_width        - [double scalar] Window size for feature computation (s).
%   loop_time_seconds   - [double scalar] Simulated loop time (s).
%   max_data_lookback   - [integer scalar] Maximum samples to look back.
%   neural_baseline     - [double vector] Baseline feature per channel.
%
% OUTPUTS:
%   neural_feature      - [double matrix] Computed neural features (channels x time).
%
% ASSUMPTIONS AND LIMITATIONS:
%   - Assumes 30 kHz sampling rate.
%   - Uses a 4th-order Butterworth high-pass filter at 750 Hz.
%
% MODIFICATION HISTORY:
%   2025-04-11  Grok  Extracted from make_rolling_power_features_zmh.
%
%==============================================================================
function neural_feature = process_neural_features(neural_data, kdf_nip_time, ...
    kernel_width, loop_time_seconds, max_data_lookback, neural_baseline)

% Filter neural data
filtered_neural_data = apply_bandpass_filter(neural_data);

% Simulate 30 Hz loop
kdf_indices_30k = ceil(kdf_nip_time - kdf_nip_time(1)) + 1;
kdf_index_diffs = zeros(size(kdf_indices_30k));
kdf_index_diffs(1:end-1) = diff(kdf_indices_30k);
kdf_index_diffs(kdf_index_diffs > max_data_lookback) = max_data_lookback;

% Initialize buffer and output
neural_buffer = zeros(size(filtered_neural_data, 2), ...
    floor(kernel_width / loop_time_seconds));
neural_feature = zeros(size(filtered_neural_data, 2), length(kdf_indices_30k));

% Compute neural features
for time_idx = 2:length(kdf_indices_30k)
    start_timestamp = kdf_indices_30k(time_idx) - kdf_index_diffs(time_idx-1);
    end_timestamp = kdf_indices_30k(time_idx) - 1;
    neural_data_subset = filtered_neural_data(start_timestamp:end_timestamp, :);
    neural_buffer = circshift(neural_buffer, [0, length(neural_data_subset)]);
    neural_buffer(:, 1:length(neural_data_subset)) = neural_data_subset';
    neural_feature(:, time_idx-1) = mean(abs(neural_buffer), 2) - neural_baseline;
end
end

%==============================================================================
% FUNCTION: apply_high_pass_filter
%
% PURPOSE:
%   Applies a 4th-order Butterworth high-pass filter to neural data.
%
% INPUTS:
%   input_data          - [double matrix] Neural data to filter (time x channels).
%
% OUTPUTS:
%   filtered_data       - [double matrix] Filtered neural data (time x channels).
%
% ASSUMPTIONS AND LIMITATIONS:
%   - Assumes 30 kHz sampling rate.
%   - Fixed cutoff frequency at 750 Hz.
%
% MODIFICATION HISTORY:
%   2025-04-11  Grok  Extracted from make_rolling_power_features_zmh.
%
%==============================================================================
function filtered_data = apply_bandpass_filter(input_data)
FS = 30e3;
[filter_b, filter_a] = butter(1, [300 1000]/(FS/2), "bandpass"); % NOTE: The order of the filter is 2n
filtered_data = input_data;
for channel_idx = 1:size(input_data, 2)
    filtered_data(:, channel_idx) = project_utils.FiltFiltM(filter_b, filter_a, ...
        input_data(:, channel_idx));
end
end
