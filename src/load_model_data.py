"""
Load and preprocess diabetes model data from raw Excel files.

This module contains functions to load time series CGM data and summary statistics
for T1DM and T2DM patients, then calculate glucose metrics (TIR, TAR, TBR) for use
in machine learning models.
"""

import pandas as pd
from pathlib import Path
import numpy as np


def load_all_time_series(data_dir: Path) -> dict:
    """
    Load time series data for all patients.

    Args:
        data_dir: Path to the data directory containing Shanghai_T1DM and Shanghai_T2DM folders

    Returns:
        Dictionary with structure: {patient_id: {'time_series': DataFrame, 'summary': DataFrame}}
    """
    t1dm_path = data_dir / 'Shanghai_T1DM_Summary.xlsx'
    t2dm_path = data_dir / 'Shanghai_T2DM_Summary.xlsx'
    t1_ts_path = data_dir / 'Shanghai_T1DM'
    t2_ts_path = data_dir / 'Shanghai_T2DM'

    # Load summary data
    t1dm_summary = pd.read_excel(t1dm_path)
    t2dm_summary = pd.read_excel(t2dm_path)

    all_ts = {}

    # Load T1DM time series
    for patient in t1dm_summary['Patient Number']:
        try:
            df = pd.read_excel(t1_ts_path / (patient + '.xlsx'))
        except FileNotFoundError:
            df = pd.read_excel(t1_ts_path / (patient + '.xls'))

        # Remove Chinese column that repeats food names
        df.drop(columns=['饮食'], inplace=True)
        all_ts[patient] = {
            'time_series': df,
            'summary': t1dm_summary[t1dm_summary['Patient Number'] == patient]
        }

    # Load T2DM time series
    for patient in t2dm_summary['Patient Number']:
        try:
            df = pd.read_excel(t2_ts_path / (patient + '.xlsx'))
        except FileNotFoundError:
            df = pd.read_excel(t2_ts_path / (patient + '.xls'))

        # Remove Chinese column that repeats food names
        if '饮食' in df.columns:
            df.drop(columns=['饮食'], inplace=True)
        elif '进食量' in df.columns:
            df.drop(columns=['进食量'], inplace=True)

        all_ts[patient] = {
            'time_series': df,
            'summary': t2dm_summary[t2dm_summary['Patient Number'] == patient]
        }

    # Clean data: drop rows with missing dates
    for patient_id, patient_data in all_ts.items():
        patient_data['time_series'].dropna(subset=['Date'], inplace=True)

    return all_ts


def extract_cgm_features(ts_df: pd.DataFrame) -> dict:
    """
    Extract statistical features from CGM time series.

    Args:
        ts_df: Time series DataFrame with 'CGM (mg / dl)' column

    Returns:
        Dictionary with CGM statistical features
    """
    cgm = ts_df['CGM (mg / dl)'].dropna()

    return {
        'cgm_mean': cgm.mean(),
        'cgm_std': cgm.std(),
        'cgm_min': cgm.min(),
        'cgm_max': cgm.max(),
        'cgm_cv': cgm.std() / cgm.mean() if cgm.mean() > 0 else 0,
        'cgm_q25': cgm.quantile(0.25),
        'cgm_q75': cgm.quantile(0.75),
        'cgm_iqr': cgm.quantile(0.75) - cgm.quantile(0.25),
    }


def extract_glucose_variability(ts_df: pd.DataFrame) -> dict:
    """
    Extract glucose variability and event-based metrics from time series.

    Args:
        ts_df: Time series DataFrame with 'CGM (mg / dl)' column and 'Date' index

    Returns:
        Dictionary with variability metrics
    """
    cgm = ts_df['CGM (mg / dl)'].dropna()

    # Rate of change (glucose excursions)
    excursions = cgm.diff().abs()

    return {
        'glucose_variability': excursions.std(),
        'n_hypo_events': int((cgm < 70).sum()),
        'n_hyper_events': int((cgm > 180).sum()),
        'rapid_changes': int((excursions > 20).sum()),
        'mean_excursion': excursions.mean(),
    }


def calculate_glucose_metrics(patient_data: dict) -> tuple:
    """
    Calculate TIR, TAR, and TBR for a single patient.

    Metrics are calculated as the mean of daily percentages, considering only
    full days (days with 24 hours of CGM data).

    Args:
        patient_data: Dictionary with 'time_series' and 'summary' keys

    Returns:
        Tuple of (tar_mean, tir_mean, tbr_mean)
    """
    ts_df = patient_data['time_series']

    # Filter to only full days (24 hours of data)
    full_dates = ts_df.groupby(ts_df['Date'].dt.date).apply(
        lambda x: len(x['Date'].dt.hour.unique()) == 24
    )
    df_filtered = ts_df[ts_df['Date'].dt.date.isin(full_dates[full_dates].index)]

    # Calculate daily percentages
    tar = df_filtered.groupby(df_filtered['Date'].dt.date)['CGM (mg / dl)'].apply(
        lambda x: (x > 180).sum() / len(x)
    )
    tir = df_filtered.groupby(df_filtered['Date'].dt.date)['CGM (mg / dl)'].apply(
        lambda x: ((x >= 70) & (x <= 180)).sum() / len(x)
    )
    tbr = df_filtered.groupby(df_filtered['Date'].dt.date)['CGM (mg / dl)'].apply(
        lambda x: (x < 70).sum() / len(x)
    )

    # Sanity check: all percentages should sum to ~1.0
    df_check = pd.concat([tar, tir, tbr], axis=1, keys=['tar', 'tir', 'tbr'])
    total_perc = df_check['tar'] + df_check['tir'] + df_check['tbr']
    if not np.isclose(total_perc, 1.0).all():
        raise ValueError(f"Percentage calculation error for patient")

    return tar.mean(), tir.mean(), tbr.mean()


def load_model_data(data_dir: Path = None) -> pd.DataFrame:
    """
    Load and preprocess diabetes model data.

    Loads time series CGM data and summary statistics for all patients,
    calculates glucose metrics (TIR, TAR, TBR), and combines with HbA1c values
    and patient demographics.

    Args:
        data_dir: Path to the data directory. If None, uses PROJECT_ROOT/data/processed/data

    Returns:
        DataFrame with columns:
        - patient: Patient ID
        - tar: Time Above Range (mean percentage, values > 180 mg/dL)
        - tir: Time In Range (mean percentage, values 70-180 mg/dL)
        - tbr: Time Below Range (mean percentage, values < 70 mg/dL)
        - hba1c: HbA1c glycated hemoglobin (mmol/mol)
        - age: Age in years
        - gender: Gender (0=Female, 1=Male)
        - bmi: Body Mass Index (kg/m2)
        - height: Height in meters
        - weight: Weight in kilograms
        - smoking_history: Smoking history in pack years
        - alcohol_drinker: Binary flag (1=drinker, 0=non-drinker)
        - diabetes_type: Type of diabetes (0=T1DM, 1=T2DM)
        - diabetes_duration: Duration of diabetes in years
        - cgm_mean: Mean CGM reading (mg/dL)
        - cgm_std: Standard deviation of CGM readings
        - cgm_min: Minimum CGM reading
        - cgm_max: Maximum CGM reading
        - cgm_cv: Coefficient of variation (std/mean)
        - cgm_q25: 25th percentile of CGM
        - cgm_q75: 75th percentile of CGM
        - cgm_iqr: Interquartile range of CGM
        - glucose_variability: Standard deviation of glucose excursions (rate of change)
        - n_hypo_events: Number of hypoglycemic events (CGM < 70 mg/dL)
        - n_hyper_events: Number of hyperglycemic events (CGM > 180 mg/dL)
        - rapid_changes: Number of rapid glucose changes (>20 mg/dL)
        - mean_excursion: Mean magnitude of glucose excursions

        Rows with missing or invalid HbA1c values ('/') are excluded.
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.resolve()
        data_dir = project_root / 'data' / 'processed' / 'data'

    # Load all time series data
    all_ts = load_all_time_series(data_dir)

    # Calculate metrics for each patient
    data = []
    for patient_id, patient_data in all_ts.items():
        tar_mean, tir_mean, tbr_mean = calculate_glucose_metrics(patient_data)
        ts_df = patient_data['time_series']
        summary = patient_data['summary'].iloc[0]

        # Extract CGM features from time series
        cgm_features = extract_cgm_features(ts_df)
        variability_features = extract_glucose_variability(ts_df)

        # Extract and encode features
        hba1c_value = summary['HbA1c (mmol/mol)']

        # Gender: convert 1,2 to 0,1 (0=Female, 1=Male)
        gender = summary['Gender (Female=1, Male=2)'] - 1

        # Alcohol: convert to binary (1=drinker, 0=non-drinker)
        alcohol = 1 if 'drinker' in str(summary['Alcohol Drinking History (drinker/non-drinker)']).lower() else 0

        # Diabetes duration: standardize column name
        if 'Duration of Diabetes  (years)' in summary.index:
            diabetes_duration = summary['Duration of Diabetes  (years)']
        else:
            diabetes_duration = summary['Duration of diabetes (years)']

        # Diabetes type: convert to binary (0=T1DM, 1=T2DM)
        diabetes_type = 0 if summary['Type of Diabetes'] == 'T1DM' else 1

        row = {
            'patient': patient_id,
            'tar': tar_mean,
            'tir': tir_mean,
            'tbr': tbr_mean,
            'hba1c': hba1c_value,
            'age': summary['Age (years)'],
            'gender': gender,
            'bmi': summary['BMI (kg/m2)'],
            'height': summary['Height (m)'],
            'weight': summary['Weight (kg)'],
            'smoking_history': summary['Smoking History (pack year)'],
            'alcohol_drinker': alcohol,
            'diabetes_type': diabetes_type,
            'diabetes_duration': diabetes_duration
        }

        # Add CGM features
        row.update(cgm_features)
        row.update(variability_features)

        data.append(row)

    model_data = pd.DataFrame(data)

    # Remove rows with invalid HbA1c values
    model_data = model_data[model_data['hba1c'] != '/'].reset_index(drop=True)

    # Convert hba1c to numeric if needed
    model_data['hba1c'] = pd.to_numeric(model_data['hba1c'], errors='coerce')

    return model_data


def get_patient_hba1c_cgm(data_dir: Path = None) -> dict:
    """
    Map each patient ID to its HbA1c reading and CGM readings.

    Args:
        data_dir: Path to the data directory. If None, uses PROJECT_ROOT/data/processed/data

    Returns:
        Dictionary {patient_id: {'hba1c': float, 'cgm': np.ndarray of CGM readings (mg/dL)}}
        Patients with missing HbA1c ('/') are excluded.
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.resolve()
        data_dir = project_root / 'data' / 'processed' / 'data'

    all_ts = load_all_time_series(data_dir)

    result = {}
    for patient_id, patient_data in all_ts.items():
        hba1c = pd.to_numeric(patient_data['summary']['HbA1c (mmol/mol)'].iloc[0], errors='coerce')
        if pd.isna(hba1c):
            continue
        result[patient_id] = {
            'hba1c': float(hba1c),
            'cgm': patient_data['time_series']['CGM (mg / dl)'].dropna().to_numpy()
        }

    return result


if __name__ == '__main__':
    # Example usage
    df = load_model_data()
    print(f"Loaded {len(df)} patient records")
    print(df.head())
    print(df.info())
