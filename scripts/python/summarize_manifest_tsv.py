from unicodedata import category

import yaml
from scipy.stats import false_discovery_control


def summarize_manifest(file_path="../../reports/manifest_annotations.yaml"):
    dataset_counts = {
        "Combined": {"Unidirectional": 0, "Bidirectional": 0},
        "Independent": {"Unidirectional": 0, "Bidirectional": 0},
    }

    trial_counts = {
        "Combined": {"Unidirectional": 0, "Bidirectional": 0},
        "Independent": {"Unidirectional": 0, "Bidirectional": 0},
    }

    with open (file_path, 'r') as f:
        metadata = yaml.safe_load(f)

    for job_id in metadata.values():
        kinematic_metadata = job_id.get('kinematics', {})
        for kinematic_info in kinematic_metadata.values():
            kinematic_type = kinematic_info.get("type", "").capitalize()
            gestures = kinematic_info.get("gestures", {})

            directionality = "Bidirectional" if len(gestures) >=2 else "Unidirectional"

            if kinematic_type in dataset_counts:
                dataset_counts[kinematic_type][directionality] += 1

                total_trials = min(gestures.values())
                trial_counts[kinematic_type][directionality] += total_trials

    # 4. Format and print the first table (Dataset Counts)
    print("--- Summary by Dataset Count ---")
    header1 = "|             | Unidirectional | Bidirectional |"
    separator1 = "|-------------|----------------|---------------|"
    combined_row1 = f"| Combined    | {dataset_counts['Combined']['Unidirectional']:<14} | {dataset_counts['Combined']['Bidirectional']:<13} |"
    independent_row1 = f"| Independent | {dataset_counts['Independent']['Unidirectional']:<14} | {dataset_counts['Independent']['Bidirectional']:<13} |"
    print("\n".join([header1, separator1, combined_row1, independent_row1]))

    # 5. Format and print the second table (Trial Counts)
    print("\n--- Summary by Total Trial Count ---")
    header2 = "|             | Unidirectional | Bidirectional |"
    separator2 = "|-------------|----------------|---------------|"
    combined_row2 = f"| Combined    | {trial_counts['Combined']['Unidirectional']:<14} | {trial_counts['Combined']['Bidirectional']:<13} |"
    independent_row2 = f"| Independent | {trial_counts['Independent']['Unidirectional']:<14} | {trial_counts['Independent']['Bidirectional']:<13} |"
    print("\n".join([header2, separator2, combined_row2, independent_row2]))

def classify_datasets(file_path="../../reports/manifest_annotations.yaml"):
    TASKA_8_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd10', 'd12'}
    DEKA_6_SET = {'d1', 'd2', 'd3', 'd10', 'd12'}
    FINGERS_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'}

    category_counts = {
        "TASKA-8": 0,
        "DEKA-6": 0,
        "ETD2-RF": 0,
        "ETD2-R": 0,
        "ETD2": 0,
        "Unknown": 0,
    }

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    for job_id in data.values():
        kinematics_present = set(job_id.get('kinematics', {}).keys())
        is_categorized = False

        if TASKA_8_SET.issubset(kinematics_present):
            category_counts["TASKA-8"] +=1
            is_categorized = True

        if DEKA_6_SET.issubset(kinematics_present):
            category_counts["DEKA-6"] += 1
            is_categorized = True

        if FINGERS_SET.intersection(kinematics_present) and 'd10' in kinematics_present and 'd12' in kinematics_present:
            category_counts["ETD2-RF"] += 1
            is_categorized = True

        if FINGERS_SET.intersection(kinematics_present) and 'd12' in kinematics_present:
            category_counts["ETD2-R"] += 1
            is_categorized = True

        if FINGERS_SET.intersection(kinematics_present):
            category_counts["ETD2"] += 1
            is_categorized = True

        if not is_categorized:
            category_counts["Unknown"] += 1

    print("\n--- Summary by Dataset Category (Inclusive Count) ---")
    print("| Category      | Dataset Count |")
    print("|---------------|---------------|")
    for category_name, count in category_counts.items():
        print(f"| {category_name:<13} | {count:<13} |")

def classify_bidir_indep_datasets(file_path="../../reports/manifest_annotations.yaml"):
    TASKA_8_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd10', 'd12'}
    DEKA_6_SET = {'d1', 'd2', 'd3', 'd10', 'd12'}
    FINGERS_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'}

    category_counts = {
        "TASKA-8": 0,
        "DEKA-6": 0,
        "ETD2-RF": 0,
        "ETD2-R": 0,
        "ETD2": 0,
        "Unknown": 0,
    }

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # 4. Iterate through each job, filter its kinematics, then classify
    for job_data in data.values():
        job_kinematics_data = job_data.get('kinematics', {})

        # --- NEW LOGIC: Create a set of only independent & bidirectional kinematics ---
        indep_bidir_kinematics = set()
        for k_name, k_info in job_kinematics_data.items():
            is_independent = k_info.get('type') == 'independent'
            is_bidirectional = len(k_info.get('gestures', {})) >= 2
            if is_independent and is_bidirectional:
                indep_bidir_kinematics.add(k_name)

        # Now, run classification checks against this filtered set
        is_categorized = False
        if TASKA_8_SET.issubset(indep_bidir_kinematics):
            category_counts["TASKA-8"] += 1
            is_categorized = True
        if DEKA_6_SET.issubset(indep_bidir_kinematics):
            category_counts["DEKA-6"] += 1
            is_categorized = True
        if 'd10' in indep_bidir_kinematics and 'd12' in indep_bidir_kinematics and FINGERS_SET.intersection(
                indep_bidir_kinematics):
            category_counts["ETD2-RF"] += 1
            is_categorized = True
        if 'd12' in indep_bidir_kinematics and FINGERS_SET.intersection(indep_bidir_kinematics):
            category_counts["ETD2-R"] += 1
            is_categorized = True
        if FINGERS_SET.intersection(indep_bidir_kinematics):
            category_counts["ETD2"] += 1
            is_categorized = True

        if not is_categorized:
            category_counts["Unknown"] += 1

    # 5. Format and print the final summary table
    print("--- Summary by Constrained Category (Independent & Bidirectional Only) ---")
    print("| Category      | Dataset Count |")
    print("|---------------|---------------|")
    for category_name, count in category_counts.items():
        print(f"| {category_name:<13} | {count:<13} |")

if __name__ == "__main__":
    summarize_manifest()
    classify_datasets()
    classify_bidir_indep_datasets()