function process_shared_data(varargin)
p = inputParser;
addParameter(p, 'data_root', '', @isstring);
addParameter(p, 'session_dir', '', @isstring);
addParameter(p, 'training_filename', '', @isstring);
addParameter(p, 'events_filename', '', @isstring);
addParameter(p, 'ouput_kinematics_filename', '', @isstring);
addParameter(p, 'output_events_filename', '', @isstring);
parse(p, varargin{:})
args = p.Results;

session_path = char(fullfile(args.data_root, args.session_dir));

trial_struct_training = unrl_utils.parseKEF_jag(fullfile(session_path, args.events_filename));

[kdf_kinematics_training, ~, ~, ~, kdf_nip_time_training] = unrl_utils.readKDF_jag(fullfile(session_path, args.training_filename));

if exist(args.output_kinematics_filename, 'file'), delete(args.output_kinematics_filename); end
h5create(args.output_kinematics_filename, '/kinematics', size(kdf_kinematics_training));
h5write(args.output_kinematics_filename, '/kinematics', kdf_kinematics_training);
h5create(args.output_kinematics_file, '/nip_time', size(kdf_nip_time_training));
h5write(args.output_kinematics_file, '/nip_time', kdf_nip_time_training);

if exist(args.ouput_events_filename,  'file'), delete(args.output_events_filename); end
h5create(args.output_events_filename, '/trial_start_idxs', size(trial_struct_training.trial_starts));
h5write(args.output_events_filename, '/trial_start_idxs', trial_struct_training.trial_starts);
h5create(args.output_events_filename, '/trial_stop_idxs', size(trial_struct_training.trial_stops));
h5write(args.output_events_filename, '/trial_stop_idxs', trial_struct_training.trial_stops);

end

