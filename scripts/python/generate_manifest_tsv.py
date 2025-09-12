import os
import pandas as pd
import sqlite3
import yaml

def get_full_stream_filename(session_dir):
    session_dir_path = os.path.join(config["paths"]["data_root"], session_dir)
    try:
        ns5_files = [f for f in os.listdir(session_dir_path) if f.endswith('ns5')]
        if ns5_files:
            if len(ns5_files) > 1:
                print(f"\t\tWarning 3: More than one NS5 found in {session_dir_path}")
            return ns5_files[0]
        else:
            print(f"\tWarning 1: NS5 not found -> {session_dir_path}")
            return None
    except FileNotFoundError:
        print(f"Warning 0: Directory not found -> {session_dir_path}")
        return None

with open("config.yaml") as f:
    config = yaml.safe_load(f)

with sqlite3.connect(config["paths"]["data_manifest_db"]) as conn:
    file_list = pd.read_sql_query("SELECT participant_id, session_dir, training_filename, baseline_filename FROM files", conn)

# Sanitize session_dir to work with Unix system
file_list["session_dir"] = file_list["session_dir"].apply(
    lambda x : x.replace("\\", "/").strip("/")
)

file_list["job_id"] = file_list.index + 1
file_list["events_filename"] = file_list["training_filename"].apply(lambda x: x.replace(".kdf", ".kef"))
file_list["full_stream_filename"] = file_list["session_dir"].apply(get_full_stream_filename)

file_list = file_list[["job_id", "participant_id", "session_dir", "training_filename", "baseline_filename", "events_filename", "full_stream_filename"]]
file_list.to_csv("./reports/manifest.tsv", sep="\t", index=False)