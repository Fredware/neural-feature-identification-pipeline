rule select_features:
    """
    For each job_id x feature_set x dof_config combination:
    1. Load pre-extracted features, kinematics, and events
    2. Estimate and apply alignment lag between features and kinematics
    3. Build train/test split from trial structure
    4. Run correlation pre-selection with directional kinematic splitting
    5. Run Forward Feature Selection with Gram-Schmidt Orthogonalization
    6. Write selected channels, split metadata, and alignment metadata to JSON
    """

    input:
        features = f"{SCRATCH_ROOT}/{{job_id}}/features/{{feature_set}}.h5",
        kinematics = f"{SCRATCH_ROOT}/{{job_id}}/kinematics.h5",
        events = f"{SCRATCH_ROOT}/{{job_id}}/events.h5",
    output:
        f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/selected/{{feature_set}}.json"
    params:
        kinematic_idxs = lambda wc: DOF_CONFIGS[wc.dof_config]["kinematic_indices"],
        max_features = config["algorithms"]["feature_selection"]["max_features"],
        min_correlation = config["algorithms"]["feature_selection"]["min_correlation"],
        max_lag=config["algorithms"]["feature_selection"].get("max_lag_samples", 12),
        train_ratio = config["analysis"]["train_test_split"]["train_ratio"],
        split_type = config["analysis"]["train_test_split"]["split_type"],
        include_combined = config["analysis"]["train_test_split"]["include_combined"],
        include_rest=config["analysis"]["train_test_split"].get("include_rest", True),
    log:
        f"{RESULTS_ROOT}/logs/select_features/{{job_id}}_{{feature_set}}_{{dof_config}}.log"
    threads: 1
    resources:
        mem_mb = lambda wildcards: 32000 if wildcards.feature_set =="DWT-DB4" else 16000,
        runtime = 15,
        slurm_account = "george",
        slurm_partition = "kingspeak"
    shell:
        """
        python scripts/python/select_features.py \
        --features {input.features} \
        --kinematics {input.kinematics} \
        --events {input.events} \
        --kinematic-indices {params.kinematic_idxs} \
        --max-features {params.max_features} \
        --min-correlation {params.min_correlation} \
        --max-lag {params.max_lag} \
        --train-ratio {params.train_ratio} \
        --split-type {params.split_type} \
        --include-combined {params.include_combined} \
        --include-rest {params.include_rest} \
        --output {output} \
        > {log} 2>&1
        """

