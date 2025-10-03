rule preprocess_session:
    """
    For each job_id, save the shared kinematic labels and event markers to distinct HDF5 files.
    This rule runs once per job_id.
    """
    output:
        kinematics=f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        events=f"{SCRATCH_ROOT}/{{job_id}}/events.h5"
    log:
        f"{RESULTS_ROOT}/logs/preprocess_session/{{job_id}}.log"
    params:
        job_info=lambda wildcards: manifest.loc[int(wildcards.job_id)],
    shell:
        r"""
        mkdir -p $(dirname {output.kinematics})
        mkdir -p $(dirname {output.events})
        
        module load matlab/R2024b
        matlab -nodisplay -r " \
            addpath('scripts/matlab'); \
            process_shared_data( \
              'data_root', '{DATA_ROOT}', \
              'session_dir', '{params.job_info.session_dir}', \ 
              'training_filename', '{params.job_info.training_filename}', \
              'events_filename', '{params.job_info.events_filename}', \
              'output_kinematics_filepath', '{output.kinematics}', \
              'output_events_filepath', '{output.events}' \
            ); exit; \
        " > {log} 2>&1
        """