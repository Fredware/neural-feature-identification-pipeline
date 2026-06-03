rule evaluate_decoder_predictions:
    """
    Compute per-trial, per-DOF metrics with intended/unintended classification.
    Reads predicitions HDF5 from decode_features.py and events from preprocess_shared_data.m
    """
    input:
        predictions=f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/decode/{{feature_set}}_predictions.h5",
        events=f"{SCRATCH_ROOT}/{{job_id}}/events.h5",
    output:
        metrics=f"{SCRATCH_ROOT}/{{job_id}}/{{dof_config}}/decode/{{feature_set}}_metrics.json",
    params:
        burn_in=config["algorithms"]["decoder"].get("burn_in_samples", 10),
    log:
        f"{RESULTS_ROOT}/logs/evaluate/{{job_id}}_{{feature_set}}_{{dof_config}}.log"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=10,
        # slurm_account="george",
        # slurm_partition="kingspeak"
        slurm_account="owner-guest",
        slurm_partition="kingspeak-guest",
        slurm_extra="'--qos=kingspeak-guest'"
    shell:
        """
        python scripts/python/evaluate_decoder_predictions.py \
            --predictions {input.predictions} \
            --events {input.events} \
            --burn-in {params.burn_in} \
            --output-metrics {output.metrics} \
            > {log} 2>&1
        """

