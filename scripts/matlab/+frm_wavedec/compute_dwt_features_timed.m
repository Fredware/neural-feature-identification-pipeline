function [dwt_features, dwt_times] = compute_dwt_features_timed(kdf_nip_timestamps, neural_data, dwt_thresh, dwt_wname)
FRAME_LEN = 2^13; % Closest power of two that is less than 300 ms @ 30kHz
DWT_WAVE = dwt_wname; % DWT_WAVE = 'sym4'; % See Baldazzi et al 2020 (haar = db1 performs best and requires shorter filters). See Diedrich et al. ('db4'/'sym7' is classically used for neural signal processing)
DWT_LEVELS = 10; % Less than or equal to: wmaxlev(FRAME_LEN, DWT_WAVE)
[n_samples, n_chans] = size(neural_data);

% --- Timestamp-to-index conversion ---
% KDF NIP to 1-based indices into neural data.
kdf_idxs_in_30k = ceil(kdf_nip_timestamps - kdf_nip_timestamps(1)) + 1; % make first frame map to idx=1
kdf_idxs_in_30k = min(kdf_idxs_in_30k, n_samples); % clamp last frame to data boundary
n_frames = length(kdf_idxs_in_30k);

% --- Output pre-allocation
dwt_features = nan(n_frames, n_chans*DWT_LEVELS);
dwt_times = zeros(n_frames, 1);

% --- Inner buffer pre-allocation ---
feature_frame = zeros(n_chans, DWT_LEVELS);

% t_tqdm_start = tic; % For online use only. Remove for offline analysis

for j = 1:n_frames
    % --- Build sample window ending at sample i ---
    % When fewer than FRAME_LEN samples, zero-pad leading edge
    i = kdf_idxs_in_30k(j);
    if i < FRAME_LEN
        frame_data = [zeros(FRAME_LEN - i, n_chans, 'single'); neural_data(1:i, :)];
    else
        frame_data = neural_data(i-FRAME_LEN+1: i, :);
    end

    t_frame = tic;
    parfor chan = 1:n_chans
        [dwt_coeffs, dwt_delims] = wavedec(double(frame_data(:, chan)), DWT_LEVELS, DWT_WAVE);
        for lvl = 1:DWT_LEVELS
            det_coeffs = detcoef(dwt_coeffs, dwt_delims, lvl);
            % Hard threshold based on dwt_thresh from baseline data.
            % dwt_thresh is (n_chans x DWT_LEVELS)
            det_coeffs_th = wthresh(det_coeffs, 'h', dwt_thresh(chan, lvl));
            feature_frame(chan, lvl) = sum((det_coeffs_th).^2);
        end
    end
    % feature_frame = feature_frame./sum(feature_frame, 2);
    dwt_times(j) = toc(t_frame);

    % --- Flatten frame: levels contiguous within each channel ---
    %   [ch1_lv1, ch1_lv2, ..., ch1_lv10, ch2_lv1, ..., chN_lv10]
    % Recoverable: reshape(dwt_features(j,:), DWT_LEVELS, n_chans)'
    for ch = 1:n_chans
        start_idx = (ch - 1) * DWT_LEVELS + 1;
        dwt_features(j, start_idx:(start_idx +DWT_LEVELS -1)) = feature_frame(ch, :);
    end

    % --- Progress Display for Testing (REMOVE FOR OFFLINE ANALYSIS) ---
    % elapsed = toc(t_tqdm_start);
    % rate = j/elapsed; % estimate of frames per sec
    % remaining = (n_frames - j) / rate; % estimate of remaining time (sec)
    % pct = j/n_frames * 100;
    % bar_filled = round(pct/5);
    % bar = [repmat('=', 1, bar_filled), repmat('-', 1, 20-bar_filled)]; % 20-char bar
    % fprintf('\r  [%s] %5.1f%% frame %d/%d %.1fs elapsed ~%.0fs (%.0f min) left  ', ...
    %    bar, pct, j, n_frames, elapsed, remaining, remaining/60);
end
% fprintf('\n'); % needed so that tqdm output does not overwrite next output.
end

