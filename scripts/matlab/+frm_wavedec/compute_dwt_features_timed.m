function [dwt_features, dwt_times] = compute_dwt_features_timed(kdf_nip_timestamps, neural_data, dwt_thresh)
FRAME_LEN = 2^13; % Closest power of two that is less than 300 ms @ 30kHz
DWT_WAVE = 'db4'; % See Baldazzi et al 2020 (haar = db1 performs best and requires shorter filters). See Diedrich et al. ('db4'/'sym7' is classically used for neural signal processing)
DWT_LEVELS = 10; % Less than or equal to: wmaxlev(FRAME_LEN, DWT_WAVE)
[n_samples, n_chans] = size(neural_data);

kdf_idxs_in_30k = ceil(kdf_nip_timestamps - kdf_nip_timestamps(1)) + 1;
kdf_idxs_in_30k = min(kdf_idxs_in_30k, n_samples);
n_frames = length(kdf_idxs_in_30k);

% neural_nip_timestamps = linspace(kdf_nip_timestamps(1), kdf_nip_timestamps(end), length(neural_data));
% dwt_idxs = find(ismember(neural_nip_timestamps, kdf_nip_timestamps));
% dwt_features = nan(length(dwt_idxs), n_chans*DWT_LEVELS);

dwt_times = [];

for j = 1:length(dwt_idxs)
    i = dwt_idxs(j);
    if i < FRAME_LEN
        temp = neural_data(1:i, :);
        frame_data = [zeros(FRAME_LEN - size(temp, 1), n_chans); temp];
    else
        frame_data = neural_data(i-FRAME_LEN+1: i, :);
    end

    % feature_vec_all_chans = nan(1, n_chans*DWT_LEVELS);
    feature_vec_all_chans = {};
    tic
    for chan=1:n_chans
        [dwt_coeffs, dwt_delims] = wavedec(frame_data(:, chan), DWT_LEVELS, DWT_WAVE);
        feature_vec_chan = nan(1, DWT_LEVELS);
        for lvl=1:DWT_LEVELS
            % feature_vec_chan(lvl) = sum((detcoef(dwt_coeffs, dwt_delims, lvl)).^2);
            detcoef_th = wthresh(detcoef(dwt_coeffs, dwt_delims, lvl), 'h', dwt_thresh{chan}(lvl));
            feature_vec_chan(lvl) = sum((detcoef_th).^2);
        end
        chan_start = DWT_LEVELS*(chan-1)+1;
        chan_stop = DWT_LEVELS*chan;
        % feature_vec_all_chans(1, chan_start:chan_stop) = feature_vec_chan;
        feature_vec_all_chans{1,chan} = feature_vec_chan;
    end
    dwt_times = [dwt_times, toc];
    dwt_features(j, :) = cell2mat(feature_vec_all_chans);
end
end

