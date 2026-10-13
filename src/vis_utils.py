"""
Plotly visualizations for the diabetes EDA and modeling notebooks.

Every function returns a Plotly Figure instead of showing it, so the caller
decides whether to display it in a notebook or save it.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ONE_DAY_MS = 24 * 60 * 60 * 1000

DIABETES_STYLE = {
    'T1DM': dict(line_color='#1f77b4', fillcolor='rgba(31, 119, 180, 0.5)'),
    'T2DM': dict(line_color='#ff7f0e', fillcolor='rgba(255, 127, 14, 0.5)'),
}

# (lower bound, upper bound, fill colour, label) in mg/dL
GLUCOSE_ZONES = [
    (0, 70, 'red', 'Low'),
    (70, 180, 'green', 'Normal'),
    (180, 400, 'orange', 'High'),
]

DURATION_METRICS = [
    ('duration_hours', 'Duration (hours)'),
    ('duration_days', 'Duration (days)'),
    ('duration_full_days', 'Duration (full days)'),
]

# Shorter labels for long column names in time series plots
COLUMN_RENAMES = {
    'Non-insulin hypoglycemic agents': 'hypogl. agents',
    'CSII - bolus insulin (Novolin R, IU)': 'bolus insulin',
    'CSII - basal insulin (Novolin R, IU / H)': 'basal insulin',
    'Insulin dose - i.v.': 'insulin i.v.',
}


def _grid_position(idx, n_cols):
    """Convert a zero-based index into a 1-based (row, col) position in a grid."""
    row, col = divmod(idx, n_cols)
    return row + 1, col + 1


def feat_vs_target(df, target_col='hba1c', n_cols=3):
    """
    Scatter plot of each feature against the target, with a least-squares line.

    The Pearson r of each feature is shown in its x-axis label. Rows where the
    feature or the target is missing are skipped for that feature only.

    Args:
        df: DataFrame containing the target and the feature columns.
        target_col: Name of the target column.
        n_cols: Number of subplot columns in the grid.

    Returns:
        Plotly Figure.
    """
    features = [c for c in df.columns if c != target_col]
    n_rows = -(-len(features) // n_cols)  # ceiling division
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)
    target = pd.to_numeric(df[target_col], errors='coerce')

    for idx, feature in enumerate(features):
        row, col = _grid_position(idx, n_cols)
        x = pd.to_numeric(df[feature], errors='coerce')
        mask = x.notna() & target.notna()
        x, y = x[mask], target[mask]

        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                           marker=dict(size=6, opacity=0.6), showlegend=False),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=x_line, y=slope * x_line + intercept, mode='lines',
                           line=dict(color='red', width=2), showlegend=False),
                row=row, col=col
            )
            fig.update_xaxes(title_text=f'{feature} (r={x.corr(y):.2f})', row=row, col=col)

        fig.update_yaxes(title_text=target_col, row=row, col=col)

    for idx in range(len(features), n_rows * n_cols):
        row, col = _grid_position(idx, n_cols)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        title=f'Features vs {target_col}',
        showlegend=False
    )
    return fig


def plot_corr(df):
    """
    Heatmap of the pairwise Pearson correlation matrix, annotated with values.

    Args:
        df: DataFrame whose numeric columns are correlated.

    Returns:
        Plotly Figure.
    """
    corr = df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={'size': 10}
    ))
    fig.update_layout(width=600, height=600, title='Correlation Matrix')
    return fig


def plot_durations(df):
    """
    Violin plots of monitoring duration for T1DM and T2DM, one panel per metric.

    Args:
        df: DataFrame with the columns duration_hours, duration_days,
            duration_full_days and diabetes_type ('T1DM' or 'T2DM').

    Returns:
        Plotly Figure.
    """
    fig = make_subplots(
        rows=1, cols=len(DURATION_METRICS),
        subplot_titles=[title for _, title in DURATION_METRICS]
    )

    for col, (metric, title) in enumerate(DURATION_METRICS, start=1):
        for dtype, style in DIABETES_STYLE.items():
            values = df.loc[df['diabetes_type'] == dtype, metric]
            fig.add_trace(
                go.Violin(
                    x=[dtype] * len(values),
                    y=values,
                    name=dtype,
                    showlegend=col == 1,
                    points='all',
                    pointpos=0,
                    meanline_visible=True,
                    **style
                ),
                row=1, col=col
            )
        fig.update_yaxes(title_text=title, row=1, col=col)
        fig.update_xaxes(title_text='Diabetes Type', row=1, col=col)

    fig.update_layout(
        title='Monitoring Period Duration by Diabetes Type',
        showlegend=True,
        height=500,
        width=1500
    )
    return fig


def _shorten_name(col_name):
    """Drop the unit part of a column name, e.g. 'BMI (kg/m2)' -> 'BMI'."""
    return col_name.split('(')[0].strip() or col_name


def _is_glucose(col_name):
    return 'CGM' in col_name or 'CBG' in col_name


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def _select_columns(all_cols, include_cols, exclude_cols):
    """Apply include/exclude filters, then split the kept columns into glucose and other."""
    kept = [
        c for c in all_cols
        if (include_cols is None or c in include_cols)
        and (exclude_cols is None or c not in exclude_cols)
    ]
    glucose_cols = [c for c in kept if _is_glucose(c)]
    other_cols = [c for c in kept if not _is_glucose(c)]
    return glucose_cols, other_cols


def _add_glucose_traces(fig, ts_df, glucose_cols):
    """Add CGM (line) and CBG (markers) to row 1, with shaded glucose zones behind them."""
    if 'CGM (mg / dl)' in glucose_cols:
        fig.add_trace(
            go.Scatter(x=ts_df['Date'], y=ts_df['CGM (mg / dl)'], mode='lines',
                       name='CGM', line=dict(color='#1f77b4')),
            row=1, col=1
        )
    if 'CBG (mg / dl)' in glucose_cols:
        fig.add_trace(
            go.Scatter(x=ts_df['Date'], y=ts_df['CBG (mg / dl)'], mode='markers',
                       name='CBG', marker=dict(color='#ff7f0e', size=5)),
            row=1, col=1
        )
    for low, high, color, label in GLUCOSE_ZONES:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.2, layer='below',
                      annotation_text=label, annotation_position='right', row=1, col=1)


def _add_column_trace(fig, row, ts_df, col_name):
    """
    Plot one non-glucose column in its own row.

    Numeric columns become a scatter over time. Text columns become markers
    labelled with their value. Columns without any values get a placeholder note.
    """
    col_data = ts_df[col_name]
    present = col_data.notna()
    date_min = ts_df['Date'].min()

    if not present.any():
        fig.add_trace(
            go.Scatter(x=[date_min], y=[0], mode='markers',
                       marker=dict(opacity=0), showlegend=False),
            row=row, col=1
        )
        fig.add_annotation(
            text='No data available', x=date_min, y=0.5, showarrow=False,
            font=dict(size=12, color='gray'), row=row, col=1
        )
    elif _is_numeric(col_data):
        fig.add_trace(
            go.Scatter(x=ts_df['Date'], y=col_data, mode='markers',
                       name=col_name, marker=dict(size=5)),
            row=row, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=ts_df.loc[present, 'Date'], y=[col_name] * present.sum(),
                       mode='markers', text=col_data[present].values, name=col_name,
                       marker=dict(size=6),
                       hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'),
            row=row, col=1
        )


def plot_patient_timeseries(patient_data, patient_id, include_cols=None, exclude_cols=None):
    """
    Stacked time-series plot of one patient's CGM, CBG and other logged columns.

    The first row shows CGM and CBG over the glucose zones (if any glucose column
    is selected). Each remaining row shows one other column. All rows share the
    same date range and one tick per day.

    Args:
        patient_data: Dict with 'time_series' (DataFrame with a 'Date' column)
            and 'summary' (DataFrame with a 'Type of Diabetes' column).
        patient_id: Patient identifier used in the title.
        include_cols: Column names to keep. None keeps all of them.
        exclude_cols: Column names to drop. None drops none.

    Returns:
        Plotly Figure.

    Raises:
        ValueError: If the filters leave no columns to plot.
    """
    ts_df = patient_data['time_series'].rename(columns=COLUMN_RENAMES)
    diabetes_type = patient_data['summary']['Type of Diabetes'].values[0]
    date_min, date_max = ts_df['Date'].min(), ts_df['Date'].max()

    all_cols = [c for c in ts_df.columns if c != 'Date']
    glucose_cols, other_cols = _select_columns(all_cols, include_cols, exclude_cols)
    if not glucose_cols and not other_cols:
        raise ValueError('No columns to plot after filtering. Check include_cols and exclude_cols.')

    n_rows = len(other_cols) + (1 if glucose_cols else 0)
    first_other_row = 2 if glucose_cols else 1
    titles = (['Glucose Measurements'] if glucose_cols else []) + other_cols
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles, vertical_spacing=0.03)

    if glucose_cols:
        _add_glucose_traces(fig, ts_df, glucose_cols)
    for row, col_name in enumerate(other_cols, start=first_other_row):
        _add_column_trace(fig, row, ts_df, col_name)

    for row in range(1, n_rows + 1):
        fig.update_xaxes(
            title_text='DateTime' if row == n_rows else '',
            tickformat='%d.%m',
            tickmode='linear',
            dtick=ONE_DAY_MS,
            range=[date_min, date_max],
            row=row, col=1
        )
    if glucose_cols:
        fig.update_yaxes(title_text='Glucose', row=1, col=1)
    for row, col_name in enumerate(other_cols, start=first_other_row):
        title = _shorten_name(col_name) if _is_numeric(ts_df[col_name]) else ''
        fig.update_yaxes(title_text=title, row=row, col=1)

    fig.update_layout(
        title=f'Time Series Data for Patient {patient_id}, {diabetes_type}',
        height=400 * n_rows,
        showlegend=True,
        hovermode='x unified'
    )
    return fig
