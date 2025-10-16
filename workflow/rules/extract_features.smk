rule extract_features:
    """
    For each job_id/feature_set combo, run the MATLAB script to extract features.
    """
    output:
        h5=f"{SCRATCH_ROOT}/{{job_id}}/features/{{feature_set}}.h5"
    input:
        mex_file="scripts/matlab/+project_utils/FilterX.mexa64",
        kinematics=f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        events=f"{SCRATCH_ROOT}/{{job_id}}/events.h5",
        config_json="workflow/matlab_config.json"
    log:
        f"{RESULTS_ROOT}/logs/extract_features/{{job_id}}_{{feature_set}}.log"
    params:
        job_info=lambda wildcards: manifest.loc[int(wildcards.job_id)],
        data_root=DATA_ROOT
    threads: 8
    resources:
      mem_mb=120000,
      time="02:00:00",
      slurm_account="george",
      slurm_partition="kingspeak"
    shell:
        r"""
        mkdir -p $(dirname {output.h5})
        mkdir -p $(dirname {log})

        module load matlab/R2024b
        matlab -nodisplay -r " \
            addpath('scripts/matlab'); \
            addpath('scripts/matlab/+project_utils'); \
            rehash toolboxcache; \
            extract_features( 'data_root', \"{DATA_ROOT}\", 'session_dir', \"{params.job_info.session_dir}\", 'full_stream_filename', \"{params.job_info.full_stream_filename}\",  'baseline_filename', \"{params.job_info.baseline_filename}\", 'kinematics_filepath', \"{input.kinematics}\", 'events_filepath', \"{input.events}\", 'feature_set_id', \"{wildcards.feature_set}\", 'output_filepath', \"{output.h5}\", 'config_filepath', \"{input.config_json}\"); exit; \
        " > {log} 2>&1
        """
