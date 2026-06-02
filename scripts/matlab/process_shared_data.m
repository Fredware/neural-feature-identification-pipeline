function process_shared_data(varargin)
p = inputParser;
addParameter(p, 'data_root', '', @isstring);
addParameter(p, 'session_dir', '', @isstring);
addParameter(p, 'training_filename', '', @isstring);
addParameter(p, 'events_filename', '', @isstring);
addParameter(p, 'output_kinematics_filepath', '', @isstring);
addParameter(p, 'output_events_filepath', '', @isstring);
parse(p, varargin{:})
args = p.Results;

session_path = char(fullfile(args.data_root, args.session_dir));

trial_struct_training = unrl_utils.parseKEF_jag(fullfile(session_path, args.events_filename));
[kdf_kinematics_training, ~, ~, ~, kdf_nip_time_training] = unrl_utils.readKDF_jag(fullfile(session_path, args.training_filename));

% --- Snap event timestamps to closest NIP timestamps in the kdf_nip_time_training timestamp vector ---
trial_starts = interp1(kdf_nip_time_training, kdf_nip_time_training, [trial_struct_training.TargOnTS], 'nearest')';
trial_stops = interp1(kdf_nip_time_training, kdf_nip_time_training, [trial_struct_training.TrialTS], 'nearest')';

% --- Build movemement matrix metadata ---
% shape (n_trials, 12 (DOFs), 4 (n_params))
% Columns = params: [amplitude, delay, rise_time, hold_time]
% Amplitude sign encodes direction (flex:+, extend:-)
% Some sessions store truncated MvntMat with only active DOFs. So I cross-reference against kinematics to verify which DOFs are active.
n_trials = numel(trial_struct_training);
active_dofs = find(any(kdf_kinematics_training ~=0, 2));
movement_matrices = zeros(n_trials, 12, 4);
for k=1:n_trials
    mvnt = trial_struct_training(k).MvntMat;
    n_rows = size(mvnt, 1)
    if n_rows == 12
        % Regular Case
        movement_matrices(k, :, :) = trial_struct_training(k).MvntMat;
    elseif n_rows == numel(active_dofs)
        % Truncated MvntMat Case:
        % Verify rows map to consecutive DOFs starting at 1
        expected_dofs = (1:n_rows)';
        if isequal(active_dofs, expected_dofs)
            movemement_matrices(k, 1:n_rows, :) = mvnt; % zero-padding
        else
            error('process_shared_data:mvnt_mapping', ...
                'Trial %d: MvntMat has %d rows but active DOFs are [%s], not [%s]. Cannot safely zero-pad.', ...
                k, n_rows, num2str(active_dofs'), num2str(expected_dofs'));
        end
    else
        error('process_shared_data:mvnt_size', ...
            'Trial %d: MvntMat has %d rows, but %d DOFs are active in kinematics. Cannot resolve mapping.', ...
            k, n_rows, numel(active_dofs));
    end
end

kdf_kinematics_training = kdf_kinematics_training';
kdf_nip_time_training = kdf_nip_time_training';

% --- Write Kinematics ---
if exist(args.output_kinematics_filepath, 'file'), delete(args.output_kinematics_filepath); end
h5create(args.output_kinematics_filepath, '/kinematics', size(kdf_kinematics_training));
h5write(args.output_kinematics_filepath, '/kinematics', kdf_kinematics_training);
h5create(args.output_kinematics_filepath, '/nip_time', size(kdf_nip_time_training));
h5write(args.output_kinematics_filepath, '/nip_time', kdf_nip_time_training);

% --- Write Events ---
if exist(args.output_events_filepath,  'file'), delete(args.output_events_filepath); end
h5create(args.output_events_filepath, '/trial_start_idxs', size(trial_starts));
h5write(args.output_events_filepath, '/trial_start_idxs', trial_starts);
h5create(args.output_events_filepath, '/trial_stop_idxs', size(trial_stops));
h5write(args.output_events_filepath, '/trial_stop_idxs', trial_stops);
h5create(args.output_events_filepath, '/movement_matrices', size(movement_matrices));
h5write(args.output_events_filepath, '/movement_matrices', movement_matrices);

end

