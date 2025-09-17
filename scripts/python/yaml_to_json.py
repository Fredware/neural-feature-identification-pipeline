import sys
import yaml
import json

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} yaml_to_json.py <input_yaml_file> <output_json_file>")
    sys.exit(1)

input_yaml_path = sys.argv[1]
output_json_path = sys.argv[2]

with open(input_yaml_path, "r") as f_yaml:
    config_yaml_data = yaml.safe_load(f_yaml)

with open(output_json_path, "w") as f_json:
    json.dump(config_yaml_data, f_json)

print(f"Successfully converted {input_yaml_path} to {output_json_path}")