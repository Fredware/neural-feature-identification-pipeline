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