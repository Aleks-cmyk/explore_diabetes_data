"""
Helpers for turning train/test splits into model inputs.
"""

from sklearn.preprocessing import PolynomialFeatures


def get_poly_feats(X_train, X_test, degree=2):
    """
    Expand features into polynomial terms, fitted on the training data only.

    With degree 2 and two inputs (a, b), the output columns are a, b, a^2, ab, b^2.
    The bias column is not included. Prints the feature counts before and after.

    Args:
        X_train: Training features (DataFrame or 2D array).
        X_test: Test features, transformed with the transformer fitted on X_train.
        degree: Maximum polynomial degree.

    Returns:
        Tuple (X_train_poly, X_test_poly) of NumPy arrays.
    """
    print(f'Creating degree {degree} polynomial features ...')

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"Original features: {X_train.shape[1]}")
    print(f"Polynomial features: {X_train_poly.shape[1]}")

    return X_train_poly, X_test_poly


def get_Xy(splits, x_cols, y_col, degree=1):
    """
    Extract feature and target arrays for the train and test splits.

    Args:
        splits: Dictionary with 'train' and 'test' DataFrames, as returned by
            prepare_splits.load_splits().
        x_cols: List of feature column names.
        y_col: Name of the target column.
        degree: If greater than 1, expand the features with get_poly_feats.

    Returns:
        Tuple (X_train, y_train, X_test, y_test). X_train and X_test are
        DataFrames when degree is 1, and NumPy arrays otherwise. y_train and
        y_test are pandas Series.
    """
    X_train, y_train, X_test, y_test = splits['train'][x_cols], splits['train'][y_col], splits['test'][x_cols], splits['test'][y_col]
    if degree > 1:
        X_train, X_test = get_poly_feats(X_train, X_test, degree=degree)
    return X_train, y_train, X_test, y_test
