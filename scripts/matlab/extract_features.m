function run_feature_extraction(varargin)
% --- 1. Parse Name-Value Input Args
p = inputParser;
addParameter(p, 'data_root', '', @ischar);
addParameter(p, 'session_dir', '', @ischar);
addParameter(p, 'training_filename', '', @ischar);
addParameter(p, 'baseline_filename', '', @ischar);
addParameter(p, 'full_stream_filename', '', @ischar);
addParameter(p, 'feature_set', '', @ischar);
addParameter(p, 'output_file', '', @ischar);
addParameter(p, 'config_file', '', @ischar);

parse(p, varargin{:})
args = p.Results;

fprintf("Loading configuration files from %s\n", args.config_file);
config_string = fileread(args.config_file);
feature_params = jsondecode(config_string);


end