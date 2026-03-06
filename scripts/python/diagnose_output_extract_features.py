import yaml
import h5py
from pathlib import Path
import sys

CONFIG_FILE = "config.yaml"

def check_feature_files(base_path: Path):
    """
    Recursively find HDF5 files within '/features' subdirectories inside the base path and checks them for corruption.
    """
    print(f"\nSearchin for feature files in {base_path}")
    files_to_check = list(base_path.rglob("features/*.h5"))

    if not files_to_check:
        print("\tNo feature files found. Verify the 'scratch_root' path in config.yaml")
        return []

    print(f"\tFound {len(files_to_check)} feature files to verify.")

    corrupted_file_list = []
    required_keys = {'features', 'computation_times'}
    for filepath in files_to_check:
        try:
            with h5py.File(filepath, 'r') as f:
                file_keys = set(f.keys())
                if not required_keys.issubset(file_keys):
                    print(f"\tIncomplete keys in file: {filepath}")
                    corrupted_file_list.append(str(filepath))
        except OSError:
            print(f"\tFile {filepath} is corrupted.")
            corrupted_file_list.append(str(filepath))

    return corrupted_file_list

def main():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        scratch_root = Path(config["paths"]["scratch_root"])
        results_root = Path(config["paths"]["scratch_root"])
    except FileNotFoundError:
        print(f"Error:  Config file ({CONFIG_FILE}) not found", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Key {e} not found in '{CONFIG_FILE}'", file=sys.stderr)
        sys.exit(1)

    corrupted_file_list = check_feature_files(scratch_root)

    print("\n===== Summary =====")
    if corrupted_file_list:
        output_filename = "corrupted_file_list.txt"
        output_filepath = results_root / output_filename

        print(f"Found {len(corrupted_file_list)} corrupted feature files.")
        with open(output_filename, 'w') as f:
            for file in corrupted_file_list:
                f.write(f"{file}\n")
        print(f"The file list has been saved to '{output_filepath}'")
    else:
        print("No corrupted files were found\n")

if __name__ == "__main__":
    main()