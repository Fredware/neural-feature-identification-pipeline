rule extract_features:
    """
    For each job_id/feature_set combo, run the MATLAB script to extract features.
    """
    output:
        h5=f"{SCRATCH_ROOT}/{{job_id}}/features/{{feature_set}}.h5"
    input:
        kinematics=f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        events=f"{SCRATCH_ROOT}/{{job_id}}/events.h5",
        config_json="workflow/matlab_config.json"
    log:
        f"{RESULTS_ROOT}/logs/extract_features/{{job_id}}_{{feature_set}}.log"
    params:
        job_info=lambda wildcards: manifest.loc[int(wildcards.job_id)],
    shell:
        r"""
        mkdir -p $(dirname {output.h5})
        mkdir -p $(dirname {log})
        
        module load matlab/R2024b
        matlab -nodisplay -r " \
        addpath('scripts/matlab'); \
        extract_features( \
            'data_root', '{DATA_ROOT}', \
            'session_dir', '{params.job_info.session_dir}', \
            'baseline_filename', '{params.job_info.baseline_filename}', \
            'training_filename', '{params.job_info.training_filename}', \
            'full_stream_filename', '{params.job_info.full_stream_filename}', \
            'feature_set', '{wildcards.feature_set}', \
            'output_file', '{output.h5}', \
            'config_file', '{input.config_json}' \
        ); exit; \
        " > {log} 2>&1
        """