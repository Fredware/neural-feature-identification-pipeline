% SPLIT_TRIALS  Splits feature and kinematic data into training and testing sets
%   [train_mask, test_mask, status] = split_trials(features, kinematics, ...
%       trial_struct, nip_time, varargin)
%
% Purpose:
%   Splits time-series data (features and kinematics) into training and testing
%   subsets based on trial information, creating logical masks for indexing.
%   Used in neural decoding or machine learning tasks to separate data by trial
%   type and timing.
%
% Inputs:
%   features        - Matrix of EMG/neural data (rows = time, cols = features)
%   kinematics      - Matrix of hand positions (rows = time, cols = DOFs)
%   trial_struct    - Struct with trial data:
%                     * targ_on_ts: Trial start timestamps
%                     * trial_ts: Trial end timestamps
%                     * mvnt_mat: 3D movement matrix (DOFs x direction x trials)
%   nip_time        - Vector of timestamps for the dataset
%   varargin        - Optional parameters (in order):
%                     * train_combo_flag (0/1): Include combos in training? (default: 0)
%                     * test_combo_flag (0/1): Include combos in testing? (default: 1)
%                     * train_percent (0 to 1): Fraction of trials for training (default: 0.5)
%                     * training_type: 'first', 'last', or 'shuffle' (default: 'first')
%
% Outputs:
%   train_mask      - Logical vector (1 = training time points)
%   test_mask       - Logical vector (1 = testing time points)
%   status          - Struct with status message and success flag
%
% Errors:
%   Throws an error for invalid inputs (e.g., mismatched dimensions, invalid train_percent).
%
% Notes:
%   Follows NASA conventions for clarity and error handling.
%   Uses snake_case per MATLAB Style Guidelines 2.0.
%   Features and kinematics matrices have rows as time and columns as features/DOFs.
%
% Example:
%   [train_mask, test_mask, status] = split_trials(features, kinematics, ...
%       trial_struct, nip_time, 1, 1, 0.7, 'shuffle');
%
% Author: Fredi R. Mino; 2025-04-30
% Version: 1.1

function [train_mask, test_mask, test_trial_idxs, status] = split_trials(features, kinematics, ...
    trial_struct, nip_time, varargin)
  % Constants
  VALID_TRAIN_TYPES = {'first', 'last', 'shuffle'};
  DEFAULT_CONFIG = struct( ...
    'train_combo_flag', 0, ...
    'test_combo_flag', 1, ...
    'train_percent', 0.5, ...
    'training_type', 'first' ...
  );

  % Initialize status
  status = struct('success', true, 'message', '');

  % Validate inputs
  [config, status, trial_struct] = parse_inputs(features, kinematics, trial_struct, ...
      nip_time, DEFAULT_CONFIG, VALID_TRAIN_TYPES, varargin{:});
  if ~status.success
    error('split_trials:InvalidInput', status.message);
  end

  % Extract movement types and trial information
  [movement_types, trial_info] = process_movement_types(trial_struct);

  % Split trials into training and testing
  [train_indices, test_indices, test_sizes] = assign_trial_indices(trial_info, nip_time, ...
      trial_struct, config);

  % Create logical masks
  train_mask = create_logical_mask(train_indices, size(kinematics, 1));
  test_mask = create_logical_mask(test_indices, size(kinematics, 1));
  test_trial_idxs.relative_stops = cumsum(test_sizes);
  test_trial_idxs.relative_starts = cumsum(test_sizes)-test_sizes+1;

  % Log status message
  status.message = sprintf('Using %s %d%% of trials for training', ...
      config.training_type, round(config.train_percent * 100));
end

% PARSE_INPUTS  Validates and parses input arguments
%
% Purpose:
%   Validates the dimensions and types of input arguments and parses optional
%   parameters from varargin, returning a configuration struct and status.
%
% Inputs:
%   features          - Matrix of EMG/neural data (rows = time, cols = features)
%   kinematics        - Matrix of kinematic data (rows = time, cols = DOFs)
%   trial_struct      - Struct with trial information
%   nip_time          - Vector of timestamps
%   default_config    - Struct with default parameter values
%   valid_train_types - Cell array of valid training type strings
%   varargin          - Optional parameters (train_combo_flag, test_combo_flag, etc.)
%
% Outputs:
%   config            - Struct with parsed parameters
%   status            - Struct with success flag and error message (if any)
%
% Notes:
%   Ensures features, kinematics, and nip_time have consistent time dimensions
%   (number of rows for matrices, length for nip_time).
%   Validates trial_struct fields and optional parameters.
%
function [config, status, trial_struct] = parse_inputs(features, kinematics, trial_struct, ...
    nip_time, default_config, valid_train_types, varargin)
  status = struct('success', true, 'message', '');
  config = default_config;

  % Validate matrix dimensions
  if size(features, 1) ~= size(kinematics, 1) || ...
      size(features, 1) ~= length(nip_time)
    status.success = false;
    status.message = 'Features, kinematics, and nip_time must have same number of time points';
    return;
  end

  % Validate trial_struct fields
  required_fields = {'TargOnTS', 'TrialTS', 'MvntMat'};
  if ~all(isfield(trial_struct, required_fields))
      status.success = false;
      status.message = 'trial_struct missing required fields';
      return;
  else
      [trial_struct.mvnt_mat] = trial_struct.MvntMat;
      [trial_struct.trial_ts] = trial_struct.TrialTS;
      [trial_struct.targ_on_ts] = trial_struct.TargOnTS;
  end

  % Parse varargin
  if nargin > 6
    config.train_combo_flag = validate_flag(varargin{1}, default_config.train_combo_flag);
  end
  if nargin > 7
    config.test_combo_flag = validate_flag(varargin{2}, default_config.test_combo_flag);
  end
  if nargin > 8
    config.train_percent = validate_percent(varargin{3}, default_config.train_percent);
    if config.train_percent <= 0 || config.train_percent > 1
      status.success = false;
      status.message = 'train_percent must be between 0 and 1';
      return;
    end
  end
  if nargin > 9
    config.training_type = validate_training_type(varargin{4}, ...
        default_config.training_type, valid_train_types);
    if isempty(config.training_type)
      status.success = false;
      status.message = sprintf( ...
          'Invalid training_type. Must be one of: %s', ...
          strjoin(valid_train_types, ', '));
      return;
    end
  end
end

% VALIDATE_FLAG  Validates binary flag input
%
% Purpose:
%   Ensures the input is a valid binary flag (0 or 1), returning the default
%   value if invalid.
%
% Inputs:
%   input    - Input value to validate
%   default  - Default value to use if input is invalid
%
% Outputs:
%   value    - Validated binary flag (0 or 1)
%
% Notes:
%   Used for train_combo_flag and test_combo_flag validation.
%
function value = validate_flag(input, default)
  if isscalar(input) && (input == 0 || input == 1)
    value = input;
  else
    value = default;
  end
end

% VALIDATE_PERCENT  Validates percentage input
%
% Purpose:
%   Ensures the input is a valid real number, returning the default value if
%   invalid.
%
% Inputs:
%   input    - Input value to validate
%   default  - Default value to use if input is invalid
%
% Outputs:
%   value    - Validated percentage value
%
% Notes:
%   Used for train_percent validation. Range check is performed in parse_inputs.
%
function value = validate_percent(input, default)
  if isscalar(input) && isreal(input)
    value = input;
  else
    value = default;
  end
end

% VALIDATE_TRAINING_TYPE  Validates training type input
%
% Purpose:
%   Ensures the input is a valid training type string, returning the default
%   value if invalid.
%
% Inputs:
%   input           - Input string to validate
%   default         - Default training type to use if invalid
%   valid_types     - Cell array of valid training type strings
%
% Outputs:
%   value           - Validated training type string
%
% Notes:
%   Valid types are 'first', 'last', or 'shuffle'.
%
function value = validate_training_type(input, default, valid_types)
  if ischar(input) && ismember(input, valid_types)
    value = input;
  else
    value = default;
  end
end

% PROCESS_MOVEMENT_TYPES  Extracts movement types and trial information
%
% Purpose:
%   Processes trial_struct to identify unique movement types and their
%   associated trial indices, grouping trials by movement type.
%
% Inputs:
%   trial_struct    - Struct with trial information (mvnt_mat field required)
%
% Outputs:
%   movement_types  - Cell array of movement type vectors (one per trial)
%   trial_info      - Struct with:
%                     * types: Cell array of unique movement types
%                     * num_trials: Number of trials per type
%                     * trial_indices: Trial indices per type
%                     * is_combo: Binary flags indicating combination movements
%
% Notes:
%   Movement types encode degrees of freedom and direction (positive = flexion,
%   negative = extension). Combination movements involve multiple DOFs.
%
function [movement_types, trial_info] = process_movement_types(trial_struct)
  movements = cat(3, trial_struct.mvnt_mat);
  [~, ~, num_trials] = size(movements);

  movement_types = cell(num_trials, 1);
  trial_info.types = {};
  trial_info.num_trials = [];
  trial_info.trial_indices = {};
  trial_info.is_combo = [];

  type_count = 0;
  for trial_idx = 1:num_trials
    active_dofs = find(movements(:, 1, trial_idx));
    movement_types{trial_idx} = sign(movements(active_dofs, 1, trial_idx)) .* active_dofs;

    [type_idx, is_new] = find_movement_type(trial_info.types, movement_types{trial_idx});
    if is_new
      type_count = type_count + 1;
      trial_info.types{type_count} = movement_types{trial_idx};
      trial_info.num_trials(type_count) = 1;
      trial_info.trial_indices{type_count} = trial_idx;
      trial_info.is_combo(type_count) = length(movement_types{trial_idx}) > 1;
    else
      trial_info.num_trials(type_idx) = trial_info.num_trials(type_idx) + 1;
      trial_info.trial_indices{type_idx} = [trial_info.trial_indices{type_idx}, trial_idx];
    end
  end
end

% FIND_MOVEMENT_TYPE  Finds or registers a movement type
%
% Purpose:
%   Checks if a movement type exists in the list of known types, returning its
%   index or registering it as a new type.
%
% Inputs:
%   types    - Cell array of known movement types
%   entry    - Movement type vector to find
%
% Outputs:
%   type_idx - Index of the matching type (0 if not found)
%   is_new   - Logical indicating if the type is new
%
% Notes:
%   Used to group trials by movement type efficiently.
%
function [type_idx, is_new] = find_movement_type(types, entry)
  type_idx = 0;
  is_new = true;
  for idx = 1:length(types)
    if isequal(types{idx}, entry)
      type_idx = idx;
      is_new = false;
      return;
    end
  end
end

% ASSIGN_TRIAL_INDICES  Assigns trials to training or testing sets
%
% Purpose:
%   Assigns time indices to training or testing sets based on trial type,
%   training type, and configuration parameters.
%
% Inputs:
%   trial_info    - Struct with trial information (types, num_trials, etc.)
%   nip_time      - Vector of timestamps
%   trial_struct  - Struct with trial timestamps (trial_ts required)
%   config        - Struct with parameters (train_percent, training_type, etc.)
%
% Outputs:
%   train_indices - Vector of time indices for training
%   test_indices  - Vector of time indices for testing
%
% Notes:
%   Supports 'first', 'last', or 'shuffle' training types. Respects combo flags
%   for including/excluding combination movements.
%
function [train_indices, test_indices, test_sizes] = assign_trial_indices(trial_info, ...
    nip_time, trial_struct, config)
  train_indices = {};
  test_indices = {};
  test_sizes = {};
  end_times = [trial_struct.trial_ts];

  for type_idx = 1:length(trial_info.types)
    indices = trial_info.trial_indices{type_idx};
    if strcmp(config.training_type, 'shuffle')
      indices = indices(randperm(length(indices)));
    end

    num_trials = trial_info.num_trials(type_idx);
    num_train = floor(num_trials * config.train_percent);
    type_train_indices = {};
    type_test_indices = {};

    for trial_pos = 1:num_trials
      trial_idx = indices(trial_pos);
      if trial_idx > 1
          start_time = end_times(trial_idx - 1);
      else
          start_time = 0;
      end
      end_time = end_times(trial_idx);

      is_train_trial = select_trial_for_training(trial_pos, num_trials, ...
          num_train, config.training_type);
      use_for_training = is_train_trial && ...
          (config.train_combo_flag || ~trial_info.is_combo(type_idx));
      use_for_testing = config.test_combo_flag || ~trial_info.is_combo(type_idx);

      trial_indices = find(nip_time >= start_time & nip_time <= end_time);
      if use_for_training
        type_train_indices{end + 1} = trial_indices;
      elseif use_for_testing
        type_test_indices{end + 1} = trial_indices;
      end
    end

    train_indices{end + 1} = [type_train_indices{:}];
    test_indices{end + 1} = [type_test_indices{:}];
    test_sizes{end + 1} = cellfun(@(x) size(x,2), type_test_indices);
  end

  train_indices = [train_indices{:}];
  test_indices = [test_indices{:}];
  test_sizes = [test_sizes{:}];
end

% SELECT_TRIAL_FOR_TRAINING  Determines if a trial is used for training
%
% Purpose:
%   Decides whether a trial should be assigned to the training set based on
%   its position, total trials, and training type.
%
% Inputs:
%   trial_pos      - Position of the trial in the sequence
%   num_trials     - Total number of trials for the movement type
%   num_train      - Number of trials to use for training
%   training_type  - String specifying training type ('first', 'last', 'shuffle')
%
% Outputs:
%   is_train       - Logical indicating if the trial is for training
%
% Notes:
%   For 'first', selects the first num_train trials. For 'last', selects the
%   last num_train trials. For 'shuffle', assumes trials are already shuffled
%   and selects the first num_train.
%
function is_train = select_trial_for_training(trial_pos, num_trials, ...
    num_train, training_type)
  switch training_type
    case 'first'
      is_train = trial_pos <= num_train;
    case 'last'
      is_train = trial_pos > (num_trials - num_train);
    otherwise
      is_train = trial_pos <= num_train;
  end
end

% CREATE_LOGICAL_MASK  Creates a logical mask from indices
%
% Purpose:
%   Generates a logical mask with true values at specified indices, used for
%   indexing training or testing data.
%
% Inputs:
%   indices      - Vector of indices to set to true
%   mask_length  - Length of the output mask
%
% Outputs:
%   mask         - Logical vector with true at specified indices
%
% Notes:
%   Ensures the mask aligns with the time dimension of kinematics/features.
%
function mask = create_logical_mask(indices, mask_length)
  mask = false(1, mask_length);
  mask(indices) = true;
end