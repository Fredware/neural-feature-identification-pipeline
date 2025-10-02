import polars as pl
import yaml
import os

def generate_manifests_by_category(
        global_manifest_path ="../../reports/manifest.tsv",
        manifest_annotations_path = "../../reports/manifest_annotations.yaml",
        output_dir = "../../reports",
        bidirectional_only=True
):
    TASKA_8_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd10', 'd12'}
    DEKA_6_SET = {'d1', 'd2', 'd3', 'd6', 'd10', 'd12'}
    FINGERS_SET = {'d1', 'd2', 'd3', 'd4', 'd5', 'd6'}

    manifest_df = pl.read_csv(global_manifest_path, separator='\t')
    with open(manifest_annotations_path, 'r') as f:
        manifest_annotations = yaml.safe_load(f)

    categorized_jobs = {
        "8DOF-TASKA": [],
        "6DOF-DEKA": [],
        "3DOF-ETDRF": [],
        "2DOF-ETDR": [],
        "1DOF-ETD": [],
        "OTHER": []
    }

    for job_id, job_data in manifest_annotations.items():
        job_kinematics = job_data.get('kinematics', {})

        # PERFORM CLASSIFICATION BASED ON BIDIRECTIONAL KINEMATICS ONLY
        job_kinematics_under_test = set()
        if bidirectional_only:
            for kinematic_id, kinematic_info in job_kinematics.items():
                # Bidirectional kinematics have two gestures: flexion and extension
                if len(kinematic_info.get('gestures', {})) == 2:
                    job_kinematics_under_test.add(kinematic_id)
        else:
            job_kinematics_under_test = set(job_kinematics.keys())

        is_categorized = False
        if TASKA_8_SET.issubset(job_kinematics_under_test):
            categorized_jobs["8DOF-TASKA"].append(job_id)
            is_categorized = True
        if DEKA_6_SET.issubset(job_kinematics_under_test):
            categorized_jobs["6DOF-DEKA"].append(job_id)
            is_categorized = True
        if FINGERS_SET.intersection(job_kinematics_under_test) and 'd10' in job_kinematics and 'd12' in job_kinematics:
            categorized_jobs["3DOF-ETDRF"].append(job_id)
            is_categorized = True
        if FINGERS_SET.intersection(job_kinematics_under_test) and 'd12' in job_kinematics:
            categorized_jobs["2DOF-ETDR"].append(job_id)
            is_categorized = True
        if FINGERS_SET.intersection(job_kinematics_under_test):
            categorized_jobs["1DOF-ETD"].append(job_id)
            is_categorized = True
        if not is_categorized:
            categorized_jobs["OTHER"].append(job_id)

    file_prefix = "manifest_bidirectional_" if bidirectional_only else "manifest_"
    print(f"Generating manifests with mode: {'Bidirectional Only' if bidirectional_only else 'Standard'}")
    for category_name, job_ids in categorized_jobs.items():
        if not job_ids:
            continue
        filtered_df = manifest_df.filter(pl.col('job_id').is_in(job_ids))
        output_filepath = os.path.join(output_dir, f"{file_prefix}{category_name}.tsv")
        filtered_df.write_csv(output_filepath, separator='\t')

if __name__ == "__main__":
    generate_manifests_by_category()
    generate_manifests_by_category(bidirectional_only=False)