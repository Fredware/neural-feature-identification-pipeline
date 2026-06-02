rule decode_features:
    """
    For each job_id x feature_set x dof_config combination:
    1. Load pre-extracted features, kinematics, and selected channels
    2. Apply alignment and train/test split from select_features output
    3. Fit Modified Kalman Filter (MKF) on training data
    4. Predict on test data (with optional gain-threshold and latching mods)
    5. Compute per-dof RMSE and correlation
    6. Write metrics (JSON) and predicitions (HDF5)
    """
    input:
        features = f"{SCRATCH_ROOT}/{{job_id}}/features/{{feature_set}}.h5",
        kinematics = f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        events = f"{SCRATCH_ROOT}/{{job_id}}/events.h5",
        selected = f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/selected/{{feature_set}}.json",
    output:
        metrics = f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/decode/{{feature_set}}_metrics.json",
        predictions = f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/decode/{{feature_set}}_predictions.h5",
    params:
        burn_in=config["algorithms"]["decoder"].get("burn_in_samples", 15),
    log:
        f"{RESULTS_ROOT}/logs/decode/{{job_id}}_{{feature_set}}_{{dof_config}}.log"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=15,
        slurm_account="george",
        slurm_partition="kingspeak"
    shell:
        """
        python scripts/python/decode.py \
            --features {input.features} \
            --kinematics {input.kinematics} \
            --events {input.events} \
            --selected {input.selected} \
            --burn-in {params.burn_in} \
            --output-metrics {output.metrics} \
            --output-predictions {output.predictions} \
            > {log} 2>&1
        """

