#!/usr/bin/env python3
"""Add a 'category' column to manifest TSV based on manual kinematic annotations.

Reads manifest_annotations.yaml and determines the DOF category for each
session based on which independent DOFs are present.

Category rules (based on DOF slot mapping):
  - 8DOF-TASKA: has d1,d2,d3,d4,d5,d6,d10,d12 (all 8 independent)
  - 6DOF-DEKA:  has d1,d2,d3,d6,d10,d12 (no d4,d5)
  - 3DOF-ETDRF: has d1,d10,d12 (thumb + wrist only)
  - 2DOF-ETDR:  has exactly 2 independent DOFs
  - 1DOF-ETD:   has exactly 1 independent DOF
  - OTHER:      anything else

Usage:
    python categorize_manifest.py \
        --manifest reports/manifest.tsv \
        --annotations reports/manifest_annotations.yaml \
        --output reports/manifest.tsv
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


# DOF sets that define each category
CATEGORY_RULES = [
    # Check most specific first
    ("8DOF-TASKA", {"d1", "d2", "d3", "d4", "d5", "d6", "d10", "d12"}),
    ("6DOF-DEKA", {"d1", "d2", "d3", "d6", "d10", "d12"}),
    ("3DOF-ETDRF", {"d1", "d10", "d12"}),
]


def get_independent_dofs(session_annotation: dict) -> set[str]:
    """Extract the set of independent DOF names from a session annotation.

    Args:
        session_annotation: Dict from the annotations YAML for one session.

    Returns:
        Set of DOF name strings (e.g., {"d1", "d2", "d10"}).
    """
    kinematics = session_annotation.get("kinematics", {})
    independent = set()
    for dof_name, dof_info in kinematics.items():
        if isinstance(dof_info, dict) and dof_info.get("type") == "independent":
            independent.add(dof_name)
    return independent


def classify_session(independent_dofs: set[str]) -> str:
    """Determine the category for a session based on its independent DOFs.

    Args:
        independent_dofs: Set of DOF name strings.

    Returns:
        Category string (e.g., "8DOF-TASKA", "6DOF-DEKA", "OTHER").
    """
    for category, required_dofs in CATEGORY_RULES:
        if required_dofs.issubset(independent_dofs):
            return category

    n_dofs = len(independent_dofs)
    if n_dofs == 2:
        return "2DOF-ETDR"
    elif n_dofs == 1:
        return "1DOF-ETD"
    else:
        return "OTHER"


def main():
    parser = argparse.ArgumentParser(
        description="Add category column to manifest from annotations."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load
    manifest = pd.read_csv(args.manifest, sep="\t")
    with open(args.annotations) as f:
        annotations = yaml.safe_load(f)

    # Classify each job
    categories = {}
    for job_id, annotation in annotations.items():
        job_id = int(job_id)
        if annotation is None:
            categories[job_id] = "UNKNOWN"
            continue
        independent_dofs = get_independent_dofs(annotation)
        category = classify_session(independent_dofs)
        categories[job_id] = category

    # Add column
    manifest["category"] = manifest["job_id"].map(categories)

    # Report
    n_mapped = manifest["category"].notna().sum()
    n_missing = manifest["category"].isna().sum()
    print(f"Mapped {n_mapped}/{len(manifest)} sessions to categories")
    if n_missing > 0:
        missing_ids = manifest.loc[manifest["category"].isna(), "job_id"].tolist()
        print(f"Missing annotations for job_ids: {missing_ids}")
        manifest["category"] = manifest["category"].fillna("UNKNOWN")

    print("\nCategory distribution:")
    print(manifest["category"].value_counts().to_string())

    # Save
    manifest.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

