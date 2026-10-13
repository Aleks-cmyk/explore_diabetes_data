"""
Create and manage train/test splits for model development.

This module splits the model data into consistent train and test sets.
Train set is used with cross-validation during model training.
Test set is held out for final model comparison.
"""

import pandas as pd
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from load_model_data import load_model_data, CGM_PERCENTILES

PCT_COLS = [f'cgm_p{p}' for p in CGM_PERCENTILES]


def add_pca_features(train: pd.DataFrame, test: pd.DataFrame, n_components: int = 3):
    """
    Replace CGM percentile columns with PCA components fitted on the train set only.
    """
    scaler = StandardScaler().fit(train[PCT_COLS])
    pca = PCA(n_components=n_components, random_state=42).fit(scaler.transform(train[PCT_COLS]))
    pc_cols = [f'pc{i + 1}' for i in range(n_components)]

    result = []
    for df in (train, test):
        pcs = pca.transform(scaler.transform(df[PCT_COLS]))
        df = df.drop(columns=PCT_COLS).copy()
        df[pc_cols] = pcs
        result.append(df.reset_index(drop=True))
    return result


def create_and_save_splits(
    data_dir: Path = None,
    output_dir: Path = None,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Create train/test splits and save them to disk.

    The train set will be used with cross-validation during model training.
    The test set is held out for final model comparison across different models.

    Args:
        data_dir: Path to raw data directory. If None, uses default.
        output_dir: Path to save splits. If None, uses data/processed/splits
        train_ratio: Proportion for training set (default 0.8)
        test_ratio: Proportion for test set (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'test' containing DataFrames
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent.resolve()
        data_dir = project_root / 'data' / 'processed' / 'data'

    if output_dir is None:
        project_root = Path(__file__).parent.parent.resolve()
        output_dir = project_root / 'data' / 'processed' / 'splits'

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify ratios sum to 1.0
    if not abs(train_ratio + test_ratio - 1.0) < 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio}")

    # Load data
    df = load_model_data(data_dir)

    # Split into train and test
    df_train, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train, df_test = add_pca_features(df_train, df_test)

    splits = {
        'train': df_train,
        'test': df_test
    }

    # Save splits as pickle files
    for split_name, split_df in splits.items():
        filepath = output_dir / f'{split_name}_split.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(split_df, f)
        print(f"Saved {split_name} split ({len(split_df)} samples) to {filepath}")

    # Save summary statistics
    summary = {
        'train_size': len(df_train),
        'test_size': len(df_test),
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'random_state': random_state,
        'total_samples': len(df)
    }

    summary_file = output_dir / 'split_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Train/Test Split Summary\n")
        f.write("=" * 40 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\nNote: Train set is used with cross-validation.\n")
        f.write("Test set is held out for final model comparison.\n")

    print(f"\nSplit summary saved to {summary_file}")
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    return splits


def load_splits(splits_dir: Path = None) -> dict:
    """
    Load pre-saved train/test splits from disk.

    Args:
        splits_dir: Path to splits directory. If None, uses data/processed/splits

    Returns:
        Dictionary with keys 'train', 'test' containing DataFrames
    """
    if splits_dir is None:
        project_root = Path(__file__).parent.parent.resolve()
        splits_dir = project_root / 'data' / 'processed' / 'splits'

    if not splits_dir.exists():
        raise FileNotFoundError(
            f"Splits directory not found at {splits_dir}. "
            "Run create_and_save_splits() first."
        )

    splits = {}
    for split_name in ['train', 'test']:
        filepath = splits_dir / f'{split_name}_split.pkl'
        if not filepath.exists():
            raise FileNotFoundError(f"Split file not found: {filepath}")

        with open(filepath, 'rb') as f:
            splits[split_name] = pickle.load(f)

    return splits


if __name__ == '__main__':
    # Create splits on first run
    splits = create_and_save_splits()
