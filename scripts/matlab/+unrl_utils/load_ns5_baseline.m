%{
 * ----------------------------------------------------------------------------
 * Filename: load_ns5_baseline.m
 * Project:  USEA Neural Data Analysis Pipeline
 * Created:  March 26, 2025
 * Author:   Fredi R. Mino
 * Purpose:  Function to load neural data and baseline timeseries from NS5 and 
 *           KDF files, applying NIP offset corrections and scaling.
 * Version:  1.0
 * Language: MATLAB
 * Inputs:   
 *           - path: Directory path to NS5 and KDF files (string)
 *           - file: NS5 filename (string)
 *           - baseline_kdf_file: Baseline KDF filename (string)
 *           - kdf_nip_time: NIP timestamps from KDF file (array)
 * Outputs:  
 *           - neural_data: Scaled neural data from NS5 file (single precision)
 *           - neural_baseline_ns5: Scaled baseline neural data from NS5 file (single precision)
 *           - baseline_nip_time: NIP timestamps from baseline KDF file (array)
 * Notes:    Handles large timeseries datasets with dynamic channel adjustment. 
 *           Uses try-catch for NIP offset calculation to support different RecStart formats.
 * Dependencies: unrl_utils.fastNSxRead2022, unrl_utils.readKDF_jag, 
 *               project_utils.CalculateNIPOffset_bhm
 * ----------------------------------------------------------------------------
%}

function [neural_data, neural_baseline_ns5, baseline_nip_time] = load_ns5_baseline(path, file, baseline_kdf_file, kdf_nip_time)
% Default number of channels 3x(8x8)
num_chans = 192;

% Ensure inputs are character arrays
path = char(path);
file = char(file);
baseline_kdf_file = char(baseline_kdf_file);

% Construct file paths
ns5_file = fullfile(path,file);
fprintf("Loading NS5 file: %\n", ns5_file);
ns2_file = regexprep(ns5_file,'.ns5','.ns2');

% Calculate NIP offset with fallback for different RecStart formats
rec_start_file = fullfile(path, sprintf('RecStart_%s.mat', path(end-14:end)));
try
    nip_offset = project_utils.CalculateNIPOffset_bhm(ns2_file, rec_start_file);
catch e
    fprintf('Failed to calculate NIP offset with RecStart: %s\n', e.message);
    rec_start_file = fullfile(path, sprintf('Kalman_SSStruct_%s.mat', path(end-14:end))); % P2015 has different RecStart.mat file
    nip_offset = project_utils.CalculateNIPOffset_bhm(ns2_file, rec_start_file);
end

% Define the time range for NS5 data with NIP offset
range = [kdf_nip_time(1), kdf_nip_time(end)] + nip_offset;  %%% NIP Offset is the number of NS2 samples leading the KDF

% Read and scale NS5 neural data
fprintf('Reading neural data from NS5 file\n')
[ns5_header, ns5_data] = unrl_utils.fastNSxRead2022('File',ns5_file,'Range',range);
% scale factor for D2A conversion
ns5_scaling_factor = (double(ns5_header.MaxAnlgVal(1))- double(ns5_header.MinAnlgVal(1))) / ... 
                    (double(ns5_header.MaxDigVal(1)) - double(ns5_header.MinDigVal(1))); 
if size(ns5_data, 1) < 192
    num_chans = size(ns5_data, 1); % Adjust for fewer channels if necessary
end
neural_data = single(ns5_data(1:num_chans,:)').*ns5_scaling_factor; % Transpose and scale to single precision

% Read and scale baseline KDF neural data
fprintf('Reading Baseline Neural from NS5 file\n');
[~, ~, ~, ~, baseline_nip_time] = unrl_utils.readKDF_jag(fullfile(path, baseline_kdf_file));
baseline_range = [baseline_nip_time(1), baseline_nip_time(end)] + nip_offset;
[baseline_header, baseline_ns5] = unrl_utils.fastNSxRead2022('File',ns5_file,'Range',baseline_range);
baseline_ns5_scaling_factor = (double(baseline_header.MaxAnlgVal(1)) - double(baseline_header.MinAnlgVal(1))) / ...
                            (double(baseline_header.MaxDigVal(1)) - double(baseline_header.MinDigVal(1))); % scale factor for dig2analog
neural_baseline_ns5 = single(baseline_ns5(1:num_chans,:)')*baseline_ns5_scaling_factor;
end