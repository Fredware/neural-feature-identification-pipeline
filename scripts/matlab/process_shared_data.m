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

trial_starts = interp1(kdf_nip_time_training, kdf_nip_time_training, [trial_struct_training.TargOnTS], 'nearest')';
trial_stops = interp1(kdf_nip_time_training, kdf_nip_time_training, [trial_struct_training.TrialTS], 'nearest')';
kdf_kinematics_training = kdf_kinematics_training';
kdf_nip_time_training = kdf_nip_time_training';

if exist(args.output_kinematics_filepath, 'file'), delete(args.output_kinematics_filepath); end
h5create(args.output_kinematics_filepath, '/kinematics', size(kdf_kinematics_training));
h5write(args.output_kinematics_filepath, '/kinematics', kdf_kinematics_training);
h5create(args.output_kinematics_filepath, '/nip_time', size(kdf_nip_time_training));
h5write(args.output_kinematics_filepath, '/nip_time', kdf_nip_time_training);

if exist(args.output_events_filepath,  'file'), delete(args.output_events_filepath); end
h5create(args.output_events_filepath, '/trial_start_idxs', size(trial_starts));
h5write(args.output_events_filepath, '/trial_start_idxs', trial_starts);
h5create(args.output_events_filepath, '/trial_stop_idxs', size(trial_stops));
h5write(args.output_events_filepath, '/trial_stop_idxs', trial_stops);

end

