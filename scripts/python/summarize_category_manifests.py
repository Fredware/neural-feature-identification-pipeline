import polars as pl
import os
import altair as alt

def summarize_category_manifests(
        manifests_dir = "../../reports",
        file_prefix = "manifest_bidirectional_"
):
    all_counts_dfs = []
    for filename in sorted(os.listdir(manifests_dir)):
        if filename.startswith(file_prefix) and filename.endswith(".tsv"):
            category_name = filename.replace(file_prefix, "").replace(".tsv", "")
            filepath = os.path.join(manifests_dir, filename)

            manifests_df = pl.read_csv(filepath, separator='\t')
            counts_df = manifests_df.group_by("participant_id").agg(
                pl.count().alias("count")
            ).with_columns(
                pl.lit(category_name).alias("category")
            )
            all_counts_dfs.append(counts_df)
    final_df = pl.concat(all_counts_dfs).sort("category", "participant_id")

    # 3. Create the stacked bar chart with Altair
    title_mode = "Bidirectional Kinematics Only" if "bidirectional" in file_prefix else "All Kinematics"

    chart = alt.Chart(final_df).mark_bar().encode(
        # X-axis: Map the 'category' column to the x-axis. ':N' denotes Nominal (categorical) data.
        x=alt.X('category:N', title='Dataset Category', axis=alt.Axis(labelAngle=-45)),

        # Y-axis: Map the sum of the 'count' column. ':Q' denotes Quantitative data.
        y=alt.Y('sum(count):Q', title='Number of Datasets'),

        # Color/Stacking: Stack bars by 'participant_id'. This is what creates the stacks.
        color=alt.Color('participant_id:N', title='Participant ID'),

        # Tooltip: Show details on hover for interactivity.
        tooltip=['category', 'participant_id', 'count']
    ).properties(
        title=f"Dataset Distribution by Category and Participant ({title_mode})",
        width=800,
        height=400
    ).interactive()

    # 4. Save the chart to an HTML file
    output_filename = f"{file_prefix}participant_summary_chart.html"
    chart.save(output_filename)
    print(f"Chart saved to '{output_filename}'. Open this file in your web browser to view.")


if __name__ == "__main__":
    summarize_category_manifests()
    summarize_category_manifests(file_prefix="manifest_unidirectional_")