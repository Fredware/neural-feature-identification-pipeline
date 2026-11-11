rule decode_features:
    """
    For each job_id x feature_set x dof_config combination:
    1. Load pre-extracted features, kinematics, and selected channels
    2. Apply alignment and train/test split from select_features output
    3. Fit Modified Kalman Filter (MKF) on training data
    4. Predict on test data (with optional gain-threshold and latching mods)
    5. Write predictions to an HDF5 for downstream analysis
    """
    input:
        features = f"{SCRATCH_ROOT}/{{job_id}}/features/{{feature_set}}.h5",
        kinematics = f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        selected = f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/selected/{{feature_set}}.json",
    output:
        predictions = f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/decode/{{feature_set}}_predictions.h5",
    log:
        f"{RESULTS_ROOT}/logs/decode/{{job_id}}_{{feature_set}}_{{dof_config}}.log"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
        # slurm_account="george",
        # slurm_partition="kingspeak"
        slurm_account="owner-guest",
        slurm_partition="kingspeak-guest",
        slurm_extra="'--qos=kingspeak-guest'"
    shell:
        """
        python scripts/python/decode_features.py \
            --features {input.features} \
            --kinematics {input.kinematics} \
            --selected {input.selected} \
            --output-predictions {output.predictions} \
            > {log} 2>&1
        """

