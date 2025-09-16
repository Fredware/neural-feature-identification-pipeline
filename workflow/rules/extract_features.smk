rule extract_features:
    """
    For each job_id/feature_set combo, run the MATLAB script to extract features.
    """
    output:
        h5=f"{SCRATCH_ROOT}/features/{{job_id}}{{feature_set}}.h5"
    input:
        config_json="workflow/matlab_config.json"
    log:
        f"reports/logs/extract_features/{{job_id}}{{feature_set}}.log"
    params:
        job_info=lambda wildcards: manifest.loc[int(wildcards.job_id)],
    shell:
        """
        mkdir -p $(dirname {log})
        module load matlab/r2024b
        matlab -nodisplay -r " \
        addpath('scripts/matlab'); \
        extract_features( \
            'data_root', '{DATA_ROOT}', \
            'session_dir', '{params.job_info.session_dir}', \
            'training_filename', '{params.job_info.training_filename}', \
            'baseline_filename', '{params.job_info.baseline_filename}', \
            'full_stream_filename', '{params.job_info.full_stream_filename}', \
            'feature_set', '{wildcards.feature_set}', \
            'output_file', '{output.h5}', \
            'config_file', '{input.config_json}' \
        ); exit; \
        " > {log} 2>&1
        """