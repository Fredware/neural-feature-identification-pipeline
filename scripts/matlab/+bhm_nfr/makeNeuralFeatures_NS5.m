function [neural_features, wf_idx, frame_computation_times] = makeNeuralFeatures_NS5(...
    threshold, kdf_nip_timestamps, neural_training, baseline_nip_timestamps, neural_baseline)
%MAKENEURAL_FEATURES_NS5  Recreates neural firing-rate features from raw NS5 data.
%
%   Simulates the 30 Hz FeedbackDecode acquisition loop using spike
%   detection and a sliding firing-rate buffer.
%
%   INPUTS:
%       threshold               - Spike threshold as multiple of std dev (e.g. -5).
%       kdf_nip_timestamps      - NIP timestamps for the training window   (1 x N_frames).
%       neural_training         - Raw neural data, training segment        (N_samples x N_chans).
%       baseline_nip_timestamps - (Optional) NIP timestamps for baseline   (1 x M_frames).
%       neural_baseline         - (Optional) Raw neural data, baseline     (M_samples x N_chans).
%
%   OUTPUTS:
%       neural_features         - Firing-rate features                     (N_chans x N_frames-1).
%       wf_idx                  - Waveform indices from the last detection frame.
%       frame_computation_times - Per-frame detection wall time            (1 x N_frames-1).

    arguments
        threshold               (1, 1) double
        kdf_nip_timestamps      (1, :) double
        neural_training         (:, :) {mustBeNumeric}
        baseline_nip_timestamps (1, :) double = []
        neural_baseline         (:, :) {mustBeNumeric} = []
    end

    % --- Constants ---
    KERNEL_WIDTH_SEC = 300e-3;    % 300 ms sliding window
    LOOP_TIME_SEC    = 30e-3;  % 30 ms (FBD20253USEA) simulated loop period
    MAX_LOOKBACK     = 37 * 30;

    % --- Shared filter (designed once, applied to both segments) ---
    [b_hp, a_hp] = butter(1, 750/15000, 'high');

    % --- Baseline mean firing rate (optional) ---
    has_baseline = ~isempty(baseline_nip_timestamps) && ~isempty(neural_baseline);
    if has_baseline
        baseline_filt   = filter_columns(neural_baseline, b_hp, a_hp);
        baseline_thresh = std(baseline_filt) .* threshold;
        baseline_frames = run_detection_loop(baseline_filt, baseline_nip_timestamps, ...
                              baseline_thresh, KERNEL_WIDTH_SEC, LOOP_TIME_SEC, MAX_LOOKBACK);
        baseline_mean   = mean(baseline_frames, 2);
    else
        baseline_mean   = 0;
    end

    % --- Training features ---
    training_filt   = filter_columns(neural_training, b_hp, a_hp);
    training_thresh = std(training_filt) .* threshold;
    [feature_frames, wf_idx, frame_computation_times] = run_detection_loop(...
        training_filt, kdf_nip_timestamps, training_thresh, ...
        KERNEL_WIDTH_SEC, LOOP_TIME_SEC, MAX_LOOKBACK);

    neural_features = feature_frames - baseline_mean;
end

% =========================================================================
%  Private helpers
% =========================================================================

function filtered = filter_columns(data, b, a)
%FILTER_COLUMNS  Applies a zero-phase IIR filter independently to each column.
    filtered = double(data);
    for ch = 1:size(data, 2)
        filtered(:, ch) = project_utils.FiltFiltM(b, a, filtered(:, ch));
    end
end

function [feature_matrix, wf_idx, computation_times] = run_detection_loop(...
    neural_data, nip_timestamps, spike_threshold, kernel_width_sec, loop_time_sec, max_lookback)
%RUN_DETECTION_LOOP  Simulates the 30 Hz FeedbackDecode spike-detection loop.
%
%   Returns a (N_chans x N_frames-1) matrix of mean firing rates per frame.

    n_chans  = size(neural_data, 2);
    n_frames = length(nip_timestamps);

    % Convert NIP timestamps to contiguous sample indices
    idx_30k  = ceil(nip_timestamps - nip_timestamps(1)) + 1;
    idx_diff = [0, diff(idx_30k)];
    idx_diff(idx_diff > max_lookback) = max_lookback;

    % Initialise state
    buffer_len        = floor(kernel_width_sec / loop_time_sec);
    neural_buffer     = zeros(n_chans, buffer_len);
    feature_matrix    = zeros(n_chans, n_frames - 1);
    neural_rem        = zeros(1, n_chans);
    wf_idx            = zeros(n_chans, 1);
    computation_times = zeros(1, n_frames - 1);

    for i = 2:n_frames
        start_idx  = idx_30k(i) - idx_diff(i);
        end_idx    = idx_30k(i) - 1;
        frame_data = neural_data(start_idx:end_idx, :);

        t_start = tic;
        [rates, wf_idx, neural_rem, ~] = bhm_nfr.findSpikesRealTimeMex(...
            frame_data, double(spike_threshold), neural_rem, 0);
        computation_times(i-1) = toc(t_start);

        neural_buffer      = [rates ./ loop_time_sec, neural_buffer(:, 1:end-1)];
        feature_matrix(:, i-1) = mean(neural_buffer, 2);
    end
end
% function [NeuralFeature, WfIdx, frame_computation_times] = makeNeuralFeatures_NS5(threshold, KDFNIPTime, BNIPTime, DNeural, NeuralBNS5)
% % Based on makeKDF_NS5_PTZ file, but simplified to only return recreated neural features from NS5 without analyzing KDF file.
% % makes a KDF file from an NS2 file
% % MRB 9/18/2018
% 
% % NS2 is at 1kHz so 1 second = 1000 data points
% % [file,path] = uigetfile(strcat(datasetPath,'\*.ns5'), 'Choose *.ns5 file...'); %'D:\P201701\20171017-094759\*.ns5','Choose *.ns5 file...');
% % NS5File = fullfile(path,file);
% % disp(NS5File);
% % NS2File = regexprep(NS5File,'.ns5','.ns2');
% % RecStartFile = fullfile(path, ['RecStart_', path(end-15:end-1), '.mat']);
% % NIPOffset = CalculateNIPOffset(NS2File, RecStartFile);
% % Range = [KDFNIPTime(1),KDFNIPTime(end)] + NIPOffset;  %%% NIP Offset is the number of NS2 samples leading the KDF.
% 
% 
% %% Baseline from NS5
% % load('E:\Shared drives\CNI\General\Ripple\xippmex\lfpSOSfilter20180417.mat');
% % load('E:\Shared drives\CNI\General\Ripple\xippmex\lfpSOSnotchfilter20180417.mat');
% % load('C:\Users\Bret Mecham\Lab\Data\lfpSOSfilter20180417.mat');
% % load('C:\Users\Bret Mecham\Lab\Data\lfpSOSnotchfilter20180417.mat');
% 
% % load('../src/filter_params/lfpSOSfilter20180417.mat');
% % load('../src/filter_params/lfpSOSnotchfilter20180417.mat');
% 
% % HeaderEMG = fastNSxRead('File',NS5File);
% % if HeaderEMG.ChannelCount > 128
% %     EMGChans = 193:224;
% % elseif HeaderEMG.ChannelCount > 96
% %     EMGChans = 97:128;
% % else
% %     EMGChans = 1:32;
% % end
% 
% % [BKDFFile,path] = uigetfile([path, 'BaselineData*.kdf'],'Choose Baseline *.kdf if desired, otherwise select [Cancel]'); % returns 0 when canceled
% % if  ~(BKDFFile == 0)
% %     disp('Reading Baseline Neural from NS5 file')
% %     [~,BFeatures,~,~,BNIPTime] = readKDF(fullfile(path, BKDFFile));
% %     BRange = [BNIPTime(1),BNIPTime(end)] + NIPOffset;
% %     [BHeader, BNS5] = fastNSxRead('File',NS5File,'Range',BRange);
% %     SfBNS5 = (double(BHeader.MaxAnlgVal(1))-double(BHeader.MinAnlgVal(1)))/(double(BHeader.MaxDigVal(1))-double(BHeader.MinDigVal(1))); % scale factor for dig2analog
% %     NeuralBNS5 = single(BNS5(1:192,:)')*SfBNS5;
% 
% ThreshRMS = threshold;
% Thresh = std(NeuralBNS5).*ThreshRMS;
% % Neural Features Baseline
% if strcmp(NeuralBNS5,'None')
%     NeuralBaseline = 0;
% else
%     [B,A] = butter(4,750/15000,'high');
%     for k=1:size(NeuralBNS5,2)%192
%         %         clc; disp(['Filtering ch ' num2str(k)]);
%         NeuralBNS5(:,k) = project_utils.FiltFiltM(B,A,NeuralBNS5(:,k));
%     end
%     % acqData simulator, loops at 30 Hz.  data sampled at 30kHz (or grabbed from 'high res')
%     BNeuralIdxIn30k = ceil(BNIPTime-BNIPTime(1))+1;
%     BNeuralIdxDiff = zeros(size(BNeuralIdxIn30k));
%     BNeuralIdxDiff(1:end-1) = diff(BNeuralIdxIn30k);
%     KernelWidth = 0.3; %sec (or 300ms)
%     LoopTime = 0.033; %sec (or 33ms) simulated software loop time
%     BNeuralBuffer = zeros(size(NeuralBNS5,2),floor(KernelWidth/LoopTime));
%     % Modeling the 30Hz sampling rate in FeedbackDecode
%     MaxDL = 37*30; %% Set the maximum length of data to look back if NIPTime separation is large.
%     BNeuralIdxDiff(BNeuralIdxDiff > MaxDL) = MaxDL;
%     NeuralREM = zeros(1,size(NeuralBNS5,2));
%     for i = 2:length(BNeuralIdxIn30k)
%         %         clc; disp(i/length(BNeuralIdxIn30k));
%         start_ts = (BNeuralIdxIn30k(i) - BNeuralIdxDiff(i-1));
%         end_ts = BNeuralIdxIn30k(i)-1;
%         BNeural = NeuralBNS5(start_ts:end_ts,:);
%         [NeuralRates,WfIdx,NeuralREM,Wf] = bhm_nfr.findSpikesRealTimeMex(double(BNeural),double(Thresh),NeuralREM,0); %slow (0.34)
% 
%         NeuralRates = NeuralRates./LoopTime;
%         BNeuralBuffer(:,2:end) = BNeuralBuffer(:,1:end-1);
%         BNeuralBuffer(:,1) = NeuralRates; %current firing rate for all neural indices
% 
%         NeuralBaselineKDF(:,i-1) = mean(BNeuralBuffer,2);
%     end
%     NeuralBaseline = mean(NeuralBaselineKDF,2);
% end
% %
% % else
% %     NeuralBaseline = zeros(192,1);
% % end
% 
% 
% %% Neural data coversion
% % disp('Reading Neural Data from NS5 file')
% % [HeaderNS5, DNS5] = fastNSxRead('File',NS5File,'Range',Range);
% % SfNS5 = (double(HeaderNS5.MaxAnlgVal(1))-double(HeaderNS5.MinAnlgVal(1)))/(double(HeaderNS5.MaxDigVal(1))-double(HeaderNS5.MinDigVal(1))); % scale factor for D2A conversion
% % DNeural = single((DNS5(1:192,:))').*SfNS5;
% 
% [B,A] = butter(4,750/15000,'high');
% for k=1:size(DNeural,2) %192
%     %     clc; disp(['Filtering ch ' num2str(k)]);
%     DNeural(:,k) = project_utils.FiltFiltM(B,A,DNeural(:,k));
% end
% 
% ThreshRMS = threshold;
% Thresh = std(DNeural).*ThreshRMS;
% 
% % acqData simulator, loops at 30 Hz.  data sampled at 30kHz (or grabbed from 'high res')
% KDFIdxIn30k = ceil(KDFNIPTime-KDFNIPTime(1))+1;
% KDFIdxDiff = zeros(size(KDFIdxIn30k));
% KDFIdxDiff(1:end-1) = diff(KDFIdxIn30k);
% KernelWidth = 0.3; %sec (or 300ms)
% LoopTime = 0.033; %sec (or 33ms) simulated software loop time
% NeuralBuffer = zeros(size(DNeural,2),floor(KernelWidth/LoopTime));
% 
% % Modeling the 30Hz sampling rate in FeedbackDecode
% MaxDL = 37*30; %% Set the maximum length of data to look back if NIPTime separation is large.
% KDFIdxDiff(KDFIdxDiff > MaxDL) = MaxDL;
% NeuralFeature = zeros(size(DNeural,2),length(KDFIdxIn30k));
% NeuralREM = zeros(1,size(DNeural,2));
% rasterBuffer = zeros(size(DNeural,2),1);
% 
% frame_computation_times = zeros(1, length(KDFIdxIn30k)-1);
% for i = 2:length(KDFIdxIn30k)
% 
%     start_ts = (KDFIdxIn30k(i) - KDFIdxDiff(i-1));
%     end_ts = KDFIdxIn30k(i)-1;
%     dNeural = DNeural(start_ts:end_ts,:);
% 
%     tic;
%     [NeuralRates,WfIdx,NeuralREM,Wf] = bhm_nfr.findSpikesRealTimeMex(double(dNeural),double(Thresh),NeuralREM,0); %slow (0.34)
%     B = find(WfIdx);
%     rasterBuffer = rasterBuffer + NeuralRates;
%     NeuralRates = NeuralRates./LoopTime;
% 
%     NeuralBuffer(:,2:end) = NeuralBuffer(:,1:end-1);
%     NeuralBuffer(:,1) = NeuralRates; %current firing rate for all neural indices
% 
%     NeuralFeature(:,i-1) = mean(NeuralBuffer,2) - NeuralBaseline; %%% subtract baseline too?
%     frame_computation_times(i-1) = toc;
% end

