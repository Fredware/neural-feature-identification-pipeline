import json

rule prepare_matlab_config:
    """
    Get feature extraction parameters from config.yaml since MATLAB 2024b does not support YAML.
    Write params to JSON, which MATLAB does support.
    """
    input:
        "config.yaml"
    output:
        "workflow/matlab_config.json"
    run:
        feature_params = config["feature_extraction_params"]
        with open(output[0], "w") as f:
            json.dump(feature_params, f, indent=4)

rule compile_matlab_mex:
    """
    Compiles the FilterX.c MEX file for Linux using the FilterM script. This function is equivalent to FILTFILT of 
    Matlab's Signal Processing Toolbox, but the direct processing of arrays let this function run 10% to 90% faster. 
    With the fast Mex function FilterX, FiltFiltM is 80% to 95% faster than FILTFILT and uses much less temporary memory.
    """
    output:
        "scripts/matlab/+project_utils/FilterX.mexa64"
    input:
        c_source="scripts/matlab/+project_utils/FilterX.c",
        script="scripts/matlab/+project_utils/FilterM.m"
    log:
        f"{RESULTS_ROOT}/logs/compile_matlab_mex.log"
    shell:
        r"""
        module load matlab/R2024b
        matlab -nodisplay -r " \
            addpath('scripts/matlab'); \
            project_utils.FilterM(); \
            exit; \
        " > {log} 2>&1
        """