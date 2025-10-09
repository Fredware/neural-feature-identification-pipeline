function [NeuralFeature, frame_computation_times] = makeRollingPowerFeatures_zmh(KDFNIPTime, BNIPTime, DNeural, NeuralBNS5, feature_params_mav)
% Based on makeKDF_NS5_PTZ file, but simplified to only return recreated neural features from NS5 without analyzing KDF file.
% makes a KDF file from an NS2 file
% MRB 9/18/2018

% NS2 is at 1kHz so 1 second = 1000 data points
% [file,path] = uigetfile(strcat(datasetPath,'\*.ns5'), 'Choose *.ns5 file...'); %'D:\P201701\20171017-094759\*.ns5','Choose *.ns5 file...');
% NS5File = fullfile(path,file);
% disp(NS5File);
% NS2File = regexprep(NS5File,'.ns5','.ns2');
% RecStartFile = fullfile(path, ['RecStart_', path(end-15:end-1), '.mat']);
% NIPOffset = CalculateNIPOffset(NS2File, RecStartFile);
% Range = [KDFNIPTime(1),KDFNIPTime(end)] + NIPOffset;  %%% NIP Offset is the number of NS2 samples leading the KDF.


%% Baseline from NS5

% load('../src/filter_params/lfpSOSfilter20180417.mat');
% load('../src/filter_params/lfpSOSnotchfilter20180417.mat');

% HeaderEMG = fastNSxRead('File',NS5File);
% if HeaderEMG.ChannelCount > 128
%     EMGChans = 193:224;
% elseif HeaderEMG.ChannelCount > 96
%     EMGChans = 97:128;
% else
%     EMGChans = 1:32;
% end

% [BKDFFile,path] = uigetfile([path, 'BaselineData*.kdf'],'Choose Baseline *.kdf if desired, otherwise select [Cancel]'); % returns 0 when canceled
% if  ~(BKDFFile == 0)
%     disp('Reading Baseline Neural from NS5 file')
%     [~,BFeatures,~,~,BNIPTime] = readKDF(fullfile(path, BKDFFile));
%     BRange = [BNIPTime(1),BNIPTime(end)] + NIPOffset;
%     [BHeader, BNS5] = fastNSxRead('File',NS5File,'Range',BRange);
%     SfBNS5 = (double(BHeader.MaxAnlgVal(1))-double(BHeader.MinAnlgVal(1)))/(double(BHeader.MaxDigVal(1))-double(BHeader.MinDigVal(1))); % scale factor for dig2analog
%     NeuralBNS5 = single(BNS5(1:192,:)')*SfBNS5;
KernelWidth = feature_params_mav.window_length_sec;

threshold = -5;
ThreshRMS = threshold;
Thresh = std(NeuralBNS5).*ThreshRMS;
% Neural Features Baseline
if strcmp(NeuralBNS5,'None')
    NeuralBaseline = 0; % No Baseline
else
    [B,A] = butter(4,750/15000,'high');
    for k=1:size(NeuralBNS5,2)%192
        %         clc; disp(['Filtering ch ' num2str(k)]);
        NeuralBNS5(:,k) = project_utils.FiltFiltM(B,A,NeuralBNS5(:,k));
    end
    % acqData simulator, loops at 30 Hz.  data sampled at 30kHz (or grabbed from 'high res')
    BNeuralIdxIn30k = ceil(BNIPTime-BNIPTime(1))+1;
    BNeuralIdxDiff = zeros(size(BNeuralIdxIn30k));
    BNeuralIdxDiff(1:end-1) = diff(BNeuralIdxIn30k);
    %     KernelWidth = .15; %sec (or 300ms)
    % LoopTime = 0.033/1000; %sec (or 33ms) simulated software loop time
    LOOP_TIME_SECONDS = 0.033; %sec (or 33ms) simulated software loop time
    BNeuralBuffer = zeros(size(NeuralBNS5,2),floor(KernelWidth/LOOP_TIME_SECONDS));
    % Modeling the 30Hz sampling rate in FeedbackDecode
    MaxDL = 37*30; %% Set the maximum length of data to look back if NIPTime separation is large.
    BNeuralIdxDiff(BNeuralIdxDiff > MaxDL) = MaxDL;
    NeuralREM = zeros(1,size(NeuralBNS5,2));
    for i = 2:length(BNeuralIdxIn30k)
        %         clc; disp(i/length(BNeuralIdxIn30k));
        start_ts = (BNeuralIdxIn30k(i) - BNeuralIdxDiff(i-1));
        end_ts = BNeuralIdxIn30k(i)-1;
        BNeural = NeuralBNS5(start_ts:end_ts,:);
        %         [NeuralRates] = mean(abs(double(BNeural)),1)';
        %         NeuralRates = NeuralRates./LoopTime;
        BNeuralBuffer = circshift(BNeuralBuffer,[0,length(BNeural)]);
        BNeuralBuffer(:,1:length(BNeural)) = BNeural'; %current firing rate for all neural indices

        NeuralBaselineKDF(:,i-1) = mean(abs(BNeuralBuffer),2);
    end
    NeuralBaseline = mean(NeuralBaselineKDF,2);
end
%
% else
%     NeuralBaseline = zeros(192,1);
% end


%% Neural data coversion
% disp('Reading Neural Data from NS5 file')
% [HeaderNS5, DNS5] = fastNSxRead('File',NS5File,'Range',Range);
% SfNS5 = (double(HeaderNS5.MaxAnlgVal(1))-double(HeaderNS5.MinAnlgVal(1)))/(double(HeaderNS5.MaxDigVal(1))-double(HeaderNS5.MinDigVal(1))); % scale factor for D2A conversion
% DNeural = single((DNS5(1:192,:))').*SfNS5;

[B,A] = butter(4,750/15000,'high');
for k=1:size(DNeural,2) %192
    %     clc; disp(['Filtering ch ' num2str(k)]);
    DNeural(:,k) = project_utils.FiltFiltM(B,A,DNeural(:,k));
end

ThreshRMS = threshold;
Thresh = std(DNeural).*ThreshRMS;

% acqData simulator, loops at 30 Hz.  data sampled at 30kHz (or grabbed from 'high res')
KDFIdxIn30k = ceil(KDFNIPTime-KDFNIPTime(1))+1;
KDFIdxDiff = zeros(size(KDFIdxIn30k));
KDFIdxDiff(1:end-1) = diff(KDFIdxIn30k);
% KernelWidth = .15; %sec (or 300ms)
LOOP_TIME_SECONDS = 0.033; %sec (or 33ms) simulated software loop time
NeuralBuffer = zeros(size(DNeural,2),floor(KernelWidth/LOOP_TIME_SECONDS));

% Modeling the 30Hz sampling rate in FeedbackDecode
MaxDL = 37*30; %% Set the maximum length of data to look back if NIPTime separation is large.
KDFIdxDiff(KDFIdxDiff > MaxDL) = MaxDL;
NeuralFeature = zeros(size(DNeural,2),length(KDFIdxIn30k));

frame_computation_times = zeros(1, length(KDFIdxIn30k)-1);
for i = 2:length(KDFIdxIn30k)
    %     clc; disp(i/length(KDFIdxIn30k));

    start_ts = (KDFIdxIn30k(i) - KDFIdxDiff(i-1));
    end_ts = KDFIdxIn30k(i)-1;


    dNeural = DNeural(start_ts:end_ts,:);
    %     [NeuralRates] = mean(abs(double(dNeural)),1)'; % findSpikesRealTimeMex
    %
    %     NeuralRates = NeuralRates./LoopTime;

    %     NeuralBuffer(:,2:end) = NeuralBuffer(:,1:end-1);
    NeuralBuffer = circshift(NeuralBuffer,[0,length(dNeural)]);
    NeuralBuffer(:,1:length(dNeural)) = dNeural'; %current firing rate for all neural indices
    
    tic;
    NeuralFeature(:,i-1) = mean(abs(NeuralBuffer),2) - NeuralBaseline; %%% subtract baseline too?
    frame_computation_times(i-1) = toc;
end