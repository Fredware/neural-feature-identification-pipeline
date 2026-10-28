# Neural Feature Identification for Long-Term High-Dimensional Control of Next-Generation Prosthetics

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Data analysis pipeline for identifying features from USEA neural signals that yield the highest performance and longevity

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         neural_feature_identification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── neural_feature_identification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes neural_feature_identification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Snakemake Pipeline Execution Commands

Dry run:
```bash
snakemake -np
```
Test Run (File subset for quick validation):
```{bash}
snakemake --profile workflow/profiles/slurm --config manifest_path="reports/manifest_test.tsv
```
Full Run:
```{bash}
snakemake --profile workflow/profiles/slurm
```
Re-Run incomplete jobs:
```{bash}
snakemake --profile workflow/profiles/slurm --rerun-incomplete
```
Selective run
```{bash}
snakemake --until <rule_name>
```
Ignore errors and keep running other independent files. Also trigger based on timestamps, not code changes.
```{bash}
snakemake --profile workflow/profiles/slurm/ --config manifest_path="reports/manifest_test_run_n15.tsv" --rerun-triggers mtime --keep-going --latency-wait 30 -np
```

## SLURM Helpful Commands
Check SLURM job submissions
```{bash}
squeue -u <uNID>
```

Check when they are scheduled to start:
```{bash}
squeue --start -u <uNID>
```

Check time left in interactive session
```{bash}
squeue -h -j $SLURM_JOB_ID -O TimeLeft
```

## Checking logs
```{bash}
sacct -j 23586518 --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

```{bash}
tail -75 reports/logs/preprocess_session/152.log
tail -75 reports/logs/extract_features/77_NFR.log
```

# Checking storage quotas
```{bash}
mychpc storage
ncdu
```

# Plotting metrics
```{bash}
python scripts/python/aggregate_decoder_metrics.py --scratch-root /scratch/general/vast/u1424875/neural-feature-identification-pipeline --manifest reports/categorized_manifest_test_run_n15.tsv --participant-info reports/participant_info.yaml --output reports/aggregated_decoder_metrics_n15.csv 
```

```{bash}
 python scripts/python/plot_decoder_dashboard.py --metrics reports/aggregated_decoder_metrics_n15.csv --metric rmse --filter intended --output reports/figures/dashboard_rmse_intended.html

# Crosstalk view
python scripts/python/plot_decoder_dashboard.py \
    --metrics reports/aggregate_metrics.csv \
    --metric rmse --filter unintended \
    --output reports/figures/dashboard_crosstalk.html
```

```{bash}
python scripts/python/plot_kalman_predictions.py --scratch-root /scratch/general/vast/u1424875/neural-feature-identification-pipeline/ --job-id 76 --feature-set NFR --dof-config 3DOF --output reports/figures/kalman_predictions/76_NFR_3DOF.png
```