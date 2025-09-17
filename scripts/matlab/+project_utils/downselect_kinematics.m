% DOWN_SELECT_KDF - Down-select kinematics and feature data from KDF file.
%
% Description:
%   Down-selects kinematics (sub_x) and feature (sub_z) data from a KDF file
%   based on specified degrees of freedom (DOFs). Optionally includes combo
%   movements and handles inactive DOFs via Kalman parameters.
%
% Inputs:
%   kef_file_path   - String, full path to .kef file
%   kdf_file_path   - String, full path to .kdf file
%   selected_dofs   - Array of integers, DOFs to include
%   varargin        - Optional parameters:
%                     - incl_combo_flag: Binary, 1 to include combo movements (default: 0)
%                     - kalman_mvnts: Vector of selected DOFs (default: all DOFs)
%                     - kalman_gain: Matrix of flexion/extension gains (default: ones)
%                     - selected_indices: Vector of feature indices (default: all indices)
%
% Outputs:
%   sub_x           - Kinematics data for selected DOFs
%   sub_z           - Feature data for selected indices
%   baseline_kdf    - Baseline feature data (zero movement)
%   sub_kdf_times   - NIP timestamps of selected data
%   trial_struct    - Subselected trial structure
%
% Example:
%   [sub_x, sub_z, baseline_kdf, sub_kdf_times, ts] = ...
%       down_select_kdf('kef_file.kef', 'kdf_file.kdf', [1, 2, 3], 1);
%
% Notes:
%   - Follows MATLAB Style Guidelines 2.0 and NASA conventions.
%   - Uses snake_case for variable names.
%   - Ensures no functions are shorter than 5 lines.

function [sub_x, sub_z, baseline_kdf, sub_kdf_times, trial_struct] = ...
    downselect_kinematics(kef_file_path, kdf_file_path, selected_dofs, varargin)

  % Initialize configuration
  config = initialize_config(selected_dofs, varargin{:});

  % Validate inputs
  validate_inputs(kef_file_path, kdf_file_path, config);

  % Load and initialize data
  [trial_struct, kdf_data] = load_data(kef_file_path, kdf_file_path);
  [kalman_mvnts, kalman_gain, selected_indices] = initialize_kalman_parameters(...
      kdf_data.x, config);

  % Check DOF completeness
  complete_dofs_flag = check_dof_completeness(selected_dofs, kalman_mvnts);

  % Process data if DOFs are complete and trials exist
  if complete_dofs_flag && ~isempty(trial_struct)
    [sub_x, sub_z, sub_kdf_times, trial_struct, baseline_kdf] = ...
        process_trials(trial_struct, kdf_data, kalman_mvnts, kalman_gain, ...
                       selected_indices, config);
  else
    % Return empty outputs if DOFs are incomplete or no trials
    [sub_x, sub_z, sub_kdf_times, trial_struct, baseline_kdf] = ...
        initialize_empty_outputs(trial_struct, kalman_mvnts, selected_indices);
  end

  % Post-process trial structure
  trial_struct = post_process_trial_struct(trial_struct);
end

% INITIALIZE_CONFIG - Set up default and optional parameters
function config = initialize_config(selected_dofs, varargin)
  config = struct();
  config.incl_combo_flag = 0; % Default: exclude combo movements
  config.align_method = 'standard'; % Default alignment method
  config.selected_dofs = selected_dofs;

  if isempty(selected_dofs)
    config.selected_dofs = [1, 2, 3, 6, 10, 12]; % Default DEKA 6 DOFs
  end

  % Parse optional inputs
  if nargin > 1
    config.incl_combo_flag = varargin{1};
  end
  if nargin > 2 && ~isempty(varargin{2})
    config.kalman_mvnts = varargin{2};
  else
    config.kalman_mvnts = [];
  end
  if nargin > 3 && ~isempty(varargin{3})
    config.kalman_gain = varargin{3};
  else
    config.kalman_gain = [];
  end
  if nargin > 4 && ~isempty(varargin{4})
    config.selected_indices = varargin{4};
  else
    config.selected_indices = [];
  end
end

% VALIDATE_INPUTS - Validate input parameters
function validate_inputs(kef_file_path, kdf_file_path, config)
  if ~ischar(kef_file_path) || ~exist(kef_file_path, 'file')
    error('down_select_kdf:invalid_kef_file', 'Invalid or missing KEF file: %s', ...
          kef_file_path);
  end
  if ~ischar(kdf_file_path) || ~exist(kdf_file_path, 'file')
    error('down_select_kdf:invalid_kdf_file', 'Invalid or missing KDF file: %s', ...
          kdf_file_path);
  end
  if ~isnumeric(config.selected_dofs) || any(config.selected_dofs <= 0)
    error('down_select_kdf:invalid_dofs', 'Selected DOFs must be positive integers');
  end
end

% LOAD_DATA - Load KEF and KDF data
function [trial_struct, kdf_data] = load_data(kef_file_path, kdf_file_path)
  trial_struct = parse_kef_jag(kef_file_path);
  [x, z, ~, ~, kdf_times] = read_kdf_jag(kdf_file_path);
  kdf_data = struct('x', x, 'z', z, 'kdf_times', kdf_times);
end

% INITIALIZE_KALMAN_PARAMETERS - Set up Kalman movement and gain parameters
function [kalman_mvnts, kalman_gain, selected_indices] = ...
    initialize_kalman_parameters(x, config)
  kalman_mvnts = find(sum(x, 2) ~= 0); % Active movements
  kalman_mvnts = kalman_mvnts(ismember(kalman_mvnts, config.selected_dofs));
  kalman_gain = ones(length(kalman_mvnts), 2);
  selected_indices = 1:size(x, 1);

  if ~isempty(config.kalman_mvnts)
    kalman_mvnts = config.kalman_mvnts;
  end
  if ~isempty(config.kalman_gain)
    kalman_gain = config.kalman_gain;
  end
  if ~isempty(config.selected_indices)
    selected_indices = config.selected_indices;
  end
end

% CHECK_DOF_COMPLETENESS - Verify if all selected DOFs are active
function complete_dofs_flag = check_dof_completeness(selected_dofs, kalman_mvnts)
  complete_dofs_flag = sum(ismember(selected_dofs, kalman_mvnts)) == length(selected_dofs);
end

% PROCESS_TRIALS - Process trial data and extract relevant kinematics/features
function [sub_x, sub_z, sub_kdf_times, trial_struct, baseline_kdf] = ...
    process_trials(trial_struct, kdf_data, kalman_mvnts, kalman_gain, ...
                   selected_indices, config)
  % Initialize output arrays
  sub_x = zeros(length(kalman_mvnts), size(kdf_data.x, 2));
  sub_z = zeros(length(selected_indices), size(kdf_data.x, 2));
  sub_kdf_times = zeros(size(kdf_data.kdf_times, 2), 1);
  baseline_kdf_ind = false(size(kdf_data.kdf_times, 2), length(trial_struct));

  % Align data
  [aligned_x, aligned_z] = align_training_data_jag(...
      kdf_data.x, kdf_data.z, [], config.align_method);

  cur_train_sub_ind = [0, 0];
  ts_idx = 1;
  for k = 1:length(trial_struct)
    [should_process, mvnt_idx, mvnt_sgn] = should_process_trial(...
        trial_struct(k), kalman_mvnts, kalman_gain, config.incl_combo_flag);
    if should_process
      [sub_x, sub_z, sub_kdf_times, cur_train_sub_ind, baseline_kdf_ind] = ...
          process_trial_data(trial_struct(k), kdf_data.kdf_times, aligned_x, ...
                             aligned_z, kalman_mvnts, selected_indices, ...
                             cur_train_sub_ind, baseline_kdf_ind, k);
      trial_struct(ts_idx) = trial_struct(k);
      ts_idx = ts_idx + 1;
    end
  end

  % Truncate outputs
  if cur_train_sub_ind(2) > 0
    sub_x = sub_x(:, 1:cur_train_sub_ind(2));
    sub_z = sub_z(:, 1:cur_train_sub_ind(2));
    sub_kdf_times = sub_kdf_times(1:cur_train_sub_ind(2));
  else
    sub_x = [];
    sub_z = [];
    sub_kdf_times = [];
  end

  % Compute baseline KDF
  baseline_kdf = kdf_data.z(:, sum(baseline_kdf_ind, 2) == 0);
end

% SHOULD_PROCESS_TRIAL - Determine if a trial should be processed
function [should_process, mvnt_idx, mvnt_sgn] = should_process_trial(...
    trial, kalman_mvnts, kalman_gain, incl_combo_flag)
  mvnt_idx = find(trial.MvntMat(:, 1), 1);
  should_process = false;
  mvnt_sgn = false;

  if ~isempty(mvnt_idx)
    mvnt_count = sum(abs(trial.MvntMat(:, 1)));
    if (incl_combo_flag && mvnt_count > 1) || mvnt_count <= 1
      if ismember(mvnt_idx, kalman_mvnts)
        mvnt_sgn = trial.MvntMat(mvnt_idx, 1) > 0;
        should_process = (mvnt_sgn && kalman_gain(kalman_mvnts == mvnt_idx, 1)) || ...
                         (~mvnt_sgn && kalman_gain(kalman_mvnts == mvnt_idx, 2));
      end
    end
  end
end

% PROCESS_TRIAL_DATA - Extract data for a single trial
function [sub_x, sub_z, sub_kdf_times, cur_train_sub_ind, baseline_kdf_ind] = ...
    process_trial_data(trial, kdf_times, x, z, kalman_mvnts, selected_indices, ...
                       cur_train_sub_ind, baseline_kdf_ind, trial_idx)
  cur_train_kdf_ind = kdf_times >= trial.TargOnTS & kdf_times <= trial.TrialTS;
  cur_train_sub_ind(1) = cur_train_sub_ind(2) + 1;
  cur_train_sub_ind(2) = cur_train_sub_ind(1) + sum(cur_train_kdf_ind) - 1;

  sub_x(:, cur_train_sub_ind(1):cur_train_sub_ind(2)) = x(kalman_mvnts, cur_train_kdf_ind);
  sub_z(:, cur_train_sub_ind(1):cur_train_sub_ind(2)) = z(selected_indices, cur_train_kdf_ind);
  sub_kdf_times(cur_train_sub_ind(1):cur_train_sub_ind(2)) = kdf_times(cur_train_kdf_ind);
  baseline_kdf_ind(:, trial_idx) = cur_train_kdf_ind;
end

% INITIALIZE_EMPTY_OUTPUTS - Return empty outputs for invalid cases
function [sub_x, sub_z, sub_kdf_times, trial_struct, baseline_kdf] = ...
    initialize_empty_outputs(trial_struct, kalman_mvnts, selected_indices)
  sub_x = zeros(length(kalman_mvnts), 0);
  sub_z = zeros(length(selected_indices), 0);
  sub_kdf_times = zeros(0, 1);
  baseline_kdf = [];
  trial_struct = trial_struct; % Return as-is
end

% POST_PROCESS_TRIAL_STRUCT - Enhance trial structure with additional fields
function trial_struct = post_process_trial_struct(trial_struct)
  rmv_ts = cellfun(@(c) isempty(c), {trial_struct.TargOnTS});
  trial_struct(rmv_ts) = [];

  for k = 1:length(trial_struct)
    trial_struct(k).TrialStart = trial_struct(k).TargOnTS;
    trial_struct(k).TrialEnd = trial_struct(k).TrialTS;
    movement_mat = zeros(12, 2);
    flex_idx = trial_struct(k).MvntMat(:, 1) > 0;
    ext_idx = trial_struct(k).MvntMat(:, 1) < 0;
    movement_mat(flex_idx, 1) = 1;
    movement_mat(ext_idx, 2) = 1;
    trial_struct(k).MovementMat = logical(movement_mat);
    trial_struct(k).TrainingOn = true;
  end
end