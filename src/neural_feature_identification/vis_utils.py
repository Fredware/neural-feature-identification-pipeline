import altair as alt
import numpy as np
import polars as pl

def make_tidy_norm(session_data: dict, project_config: dict) -> pl.DataFrame:
    """
    Structures the data into a tidy, long-format DataFrame for VIS with Altair.
    Transforms the data for VIS by downsampling and applying a normalization transformation to each channel.
    Assumes kinematics, nip_time and features are in (NxM) format where N is the number of samples.
    :param project_config:
    :param session_data:
    :return:
    """
    # Create initial nested DataFrame
    df_constructor = {
        'timestamps': session_data['kinematics']['nip_time'],
        'kinematics': session_data['kinematics']['kinematics'],
    }
    feature_names = project_config['analysis']['feature_sets']
    for name in feature_names:
        key_name = name.lower()
        if 'features' in session_data.get(key_name, {}):
            df_constructor[name] = session_data[key_name]['features']
    nested_df = pl.DataFrame(df_constructor)

    # Flatten timestamp column
    flat_df = nested_df.explode('timestamps')

    # Dynamically create expression to unnest each array column into individual columns
    unnested_expressions = []
    for name in ['kinematics'] + feature_names:
        col_name = name if name != 'kinematics' else 'kinematics'
        if col_name in flat_df.columns:
            list_len = flat_df.select(pl.col(col_name).arr.len().first()).item()
            if list_len is not None:
                unnested_expressions.extend(
                    [pl.col(col_name).arr.get(i).alias(f"{name}_{i+1}") for i in range(list_len)]
                )

    # Build the wide DataFrame by applying the unnesting expressions
    wide_df = flat_df.select(
        pl.col('timestamps'),
        *unnested_expressions
    )

    # Aggregate/downsample data to prevent browser memory issues
    print(f"Original data has {len(wide_df)} timestamps")
    plt_point_count = project_config['vis']['num_of_x_points']
    time_range = wide_df['timestamps'].max() - wide_df['timestamps'].min()
    resampling_interval = time_range / plt_point_count
    if resampling_interval < 1: resampling_interval = 1
    print(f"Resampling data by averaging over {resampling_interval:0.2f} timestamp units")
    wide_df_resampled = wide_df.group_by(
        (pl.col("timestamps") // resampling_interval).alias("time_bin")
    ).agg(pl.all().mean()).drop("time_bin")
    print(f"Resampled data has {len(wide_df_resampled)} timestamps")

    # Melt (wide2long/unpivot transform) the wide DataFrame into a long, tidy format
    tidy_df = wide_df_resampled.unpivot(index=['timestamps'], variable_name='feature_id', value_name='value')

    # Add a column for easy filtering ('feature_type': 'kinematics', 'nfr', ...)
    tidy_df = tidy_df.with_columns(
        pl.col('feature_id').str.split_exact(by='_', n=1).struct.field('field_0').alias('feature_type')
    )

    # Channel by channel normalization to improve heatmap visibility
    kinematics_df  = tidy_df.filter(pl.col('feature_type') == 'kinematics')
    features_df = tidy_df.filter(pl.col('feature_type') != 'kinematics')
    features_df_normalized = features_df.with_columns(
        min_val=pl.min('value').over('feature_id'),
        max_val=pl.max('value').over('feature_id')
    ).with_columns(
        range_val=(pl.col('max_val') - pl.col('min_val'))
    ).with_columns(
        norm_val=pl.when(pl.col('range_val')>0)
                 .then((pl.col('value') - pl.col('min_val')) / pl.col('range_val'))
                 .otherwise(0.0)
    ).with_columns(
        value=(pl.col('norm_val') * pl.col('max_val').sqrt()).fill_nan(0)
    ).drop('min_val', 'max_val', 'range_val', 'norm_val')
    tidy_df_transformed = pl.concat([kinematics_df, features_df_normalized])

    return tidy_df_transformed


def make_kinematics_line_plot(plt_df: pl.DataFrame, project_config: dict, x_domain: list) -> alt.Chart():
    """
    Creates a stacked line chart of the kinematic labels
    :param plt_df:
    :param project_config:
    :param x_domain:
    :return:
    """
    kinematics_df = plt_df.filter(pl.col('feature_type') == 'kinematics')
    plt_offset = project_config['vis']['kinematics_offset']
    kinematics_df_offset = kinematics_df.with_columns(
        pl.col('feature_id').str.split_exact(by='_', n=1).struct.field('field_1').cast(pl.Int32).alias('dof_id')
    ).with_columns(
        (pl.col('value') + (pl.col('dof_id')*plt_offset)).alias('plot_value')
    )
    plt_kinematics =  alt.Chart(kinematics_df_offset).mark_line().encode(
        x=alt.X('timestamps:Q', title='Time (NIP Units)', scale=alt.Scale(zero=False, domain=x_domain)),
        y=alt.Y('plot_value:Q', title='Kinematic Position (Offset)', axis=alt.Axis(labels=False, ticks=False, grid=False)),
        color=alt.Color('feature_id:N', title="DOF ID", sort=alt.EncodingSortField(field='dof_id', order='descending')),
    ).properties(
        width=1800,
        height=360,
    )
    return plt_kinematics

def make_events_raster_plot(trial_start_stamps: np.ndarray, trial_stop_stamps: np.ndarray, x_domain: list) -> alt.Chart():
    starts_df = pl.DataFrame({'timestamp': trial_start_stamps.flatten(), 'event': 'start'})
    stops_df = pl.DataFrame({'timestamp': trial_stop_stamps.flatten(), 'event': 'stop'})
    events_df = pl.concat([starts_df, stops_df])
    plt_event_markers = alt.Chart(events_df).mark_rule(strokeDash=[4, 4], size=2).encode(
        x=alt.X('timestamp:Q', scale=alt.Scale(zero=False, domain=x_domain)),
        color=alt.Color(
            'event:N',
            scale=alt.Scale(domain=['start', 'stop'], range=['green', 'red']),
        )
    ).properties(
        width=1800,
        height=36,
    )
    return plt_event_markers

def make_features_heatmap(plt_df: pl.DataFrame, feature_type: str, color_scheme: str, selected_chans: list[str] = None) -> alt.Chart:
    """

    """
    feature_data = plt_df.filter(pl.col('feature_type') == feature_type)

    if selected_chans:
        feature_data = feature_data.filter(pl.col('feature_id').is_in(selected_chans))

    return alt.Chart(feature_data).mark_rect().encode(
        x=alt.X('timestamps:Q', title='Time (NIP Units)', scale=alt.Scale(zero=False)),
        y=alt.Y('feature_id:O', title='Feature Index', sort=None, axis=alt.Axis(labels=False, ticks=False)),
        detail='feature_id:N',
    ).properties(
        title=f'{feature_type.upper()} Features Vs NIP Time',
        width=1800,
        height=720
    )

def make_features_line_plot(plt_df: pl.DataFrame, feature_type: str, selected_channels: list[str] = None, x_domain: list = None) -> alt.Chart:
    """
    Create a line chart for a given feature set with superimposed, transparent channels
    :param plt_df:
    :param feature_type:
    :param selected_channels:
    :param x_domain:
    :return:
    """
    feature_data = plt_df.filter(pl.col('feature_type') == feature_type)

    # Apply subsampling if a list of selected channels is provided. Intended for the DWT feature set
    if selected_channels:
        feature_data = feature_data.filter(pl.col('feature_id').is_in(selected_channels))

    return alt.Chart(feature_data).mark_line(opacity=0.12).encode(
        x=alt.X('timestamps:Q', title='Time (NIP Units)', scale=alt.Scale(zero=False, domain=x_domain)),
        y=alt.Y('value:Q', title=f'{feature_type.upper()} Normalized Activation', axis=alt.Axis(labels=False, ticks=False,grid=False)),
        detail='feature_id:N',
    ).properties(width=1800, height=360)