function [dwt_thresh_all] = compute_dwt_thresholds(baseline_data)
FRAME_LEN = 2^13; % Closest power of two that is less than 300 ms @ 30kHz
DWT_WAVE = 'db4'; % See Baldazzi et al 2020 (haar = db1 performs best and requires shorter filters). See Diedrich et al. ('db4'/'sym7' is classically used for neural signal processing)
DWT_LEVELS = 10; % Less than or equal to: wmaxlev(FRAME_LEN, DWT_WAVE)

[n_samps, n_chans] = size(baseline_data);
dwt_thresh_all = zeros(n_chans, DWT_LEVELS);
for chan = 1:n_chans
    [dwt_coeffs, dwt_delims] = wavedec(baseline_data(:,chan), DWT_LEVELS, DWT_WAVE);
    dwt_thresh_chan = zeros(1, DWT_LEVELS);
    
    for level = 1:DWT_LEVELS
        det_coefs = detcoef(dwt_coeffs, dwt_delims, level);
        sigma_j = median(abs(det_coefs))/0.6745; % Noise intensity for each level
        N_j = length(det_coefs);
        base_thresh = sigma_j*sqrt(log(N_j)); % Universal thresh defined in Donoho et al. 1994
        % Han et al. Threshold (DWT version?)
        switch level
            case 1
                dwt_thresh_chan(level) = base_thresh;
            case DWT_LEVELS
                dwt_thresh_chan(level) = base_thresh / sqrt(level);
            otherwise
                dwt_thresh_chan(level) = base_thresh / log(level+1);
        end
    end
    dwt_thresh_all(chan) = dwt_thresh_chan;    
end
end

