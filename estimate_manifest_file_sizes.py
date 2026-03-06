import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# IMPORTANT: Please update these paths to match your project structure.
MANIFEST_PATH = "reports/manifest.tsv"  # Path to your manifest file
DATA_ROOT = "/uufs/chpc.utah.edu/common/home/george-group1/data-usea"
OUTPUT_PLOT_PATH = "file_size_analysis.png" # Updated filename for new plot

def analyze_dataset_sizes():
    """
    Reads a project manifest, finds the associated raw data files,
    calculates their sizes, and generates a histogram and a box plot.
    """
    print(f"Starting file size analysis...")
    print(f"Reading manifest from: {MANIFEST_PATH}")

    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: Manifest file not found at '{MANIFEST_PATH}'. Please check the path.")
        return

    # Load the manifest, assuming it's a TSV file with a 'job_id' index
    try:
        manifest = pd.read_csv(MANIFEST_PATH, sep='\t', index_col='job_id')
    except Exception as e:
        print(f"ERROR: Could not read the manifest file. Details: {e}")
        return

    file_sizes_gb = []
    print(f"Found {len(manifest)} jobs in the manifest. Checking file sizes...")

    # Loop through each job defined in the manifest
    for job_id, row in manifest.iterrows():
        try:
            session_dir = row['session_dir']
            filename = row['training_filename']

            # Construct the full, absolute path to the data file
            full_path = os.path.join(DATA_ROOT, session_dir, filename)

            if os.path.exists(full_path):
                # Get size in bytes and convert to gigabytes
                size_bytes = os.path.getsize(full_path)
                size_gb = size_bytes / (1024**3)
                file_sizes_gb.append(size_gb)
            else:
                print(f"  [WARNING] File not found for job_id {job_id}: {full_path}")

        except KeyError as e:
            print(f"  [ERROR] Manifest is missing a required column: {e}. Skipping job_id {job_id}.")
            continue

    if not file_sizes_gb:
        print("\nAnalysis complete, but no valid data files were found to analyze.")
        return

    # --- Plotting Both a Histogram and a Box Plot ---
    print("\nGenerating plots...")
    plt.style.use('ggplot')
    # Create a figure with two subplots, arranged in 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    fig.suptitle('Distribution of Dataset File Sizes', fontsize=18)

    # --- Plot 1: Histogram ---
    ax1.hist(file_sizes_gb, bins=12, edgecolor='black', alpha=0.75)
    ax1.set_title('Frequency Histogram', fontsize=14)
    ax1.set_ylabel('Number of Datasets', fontsize=12)
    ax1.grid(axis='y', alpha=0.75)

    # Add a vertical line for the mean size to the histogram
    mean_size = np.mean(file_sizes_gb)
    ax1.axvline(mean_size, color='r', linestyle='dashed', linewidth=2)
    ax1.text(mean_size * 1.05, ax1.get_ylim()[1] * 0.9, f'Mean: {mean_size:.2f} GB', color='r')

    # --- Plot 2: Box Plot ---
    # Plotting horizontally for better label readability
    bplot = ax2.boxplot(file_sizes_gb, vert=False, whis=[5, 95], patch_artist=True)
    
    # Make the median line black and thicker to make it stand out
    median_line = bplot['medians'][0]
    median_line.set_color('black')
    median_line.set_linewidth(2)

    ax2.set_title('Quartile and Outlier Box Plot', fontsize=14)
    ax2.set_xlabel('File Size (GB)', fontsize=12)
    ax2.set_yticks([]) # Hide the y-axis ticks for a cleaner look

    # Explicitly calculate and annotate the median value with a line and text
    median_size = np.median(file_sizes_gb)
    ax2.axvline(median_size, color='b', linestyle='dotted', linewidth=2)
    ax2.text(median_size * 0.9, 0.75, f'Median: {median_size:.2f} GB', color='b', ha='right')

    # Save the entire figure to a file
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"\nSuccess! Analysis plots saved to: {OUTPUT_PLOT_PATH}")
    print("You can now copy this file to your local machine to view it.")

if __name__ == "__main__":
    analyze_dataset_sizes()