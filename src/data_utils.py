from sklearn.preprocessing import PolynomialFeatures


def get_poly_feats(X_train, X_test, degree=2):
    # Create polynomial features (degree 2: includes tar, tbr, tar², tbr², tar×tbr)
    print(f'Creating degree {degree} polynomial features ...')

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"Original features: {X_train.shape[1]}")
    print(f"Polynomial features: {X_train_poly.shape[1]}")

    return X_train_poly, X_test_poly

def get_Xy(splits, x_cols, y_col, degree=1):
    X_train, y_train, X_test, y_test = splits['train'][x_cols], splits['train'][y_col], splits['test'][x_cols], splits['test'][y_col]
    if degree > 1:
        X_train, X_test = get_poly_feats(X_train, X_test, degree=degree)
    return X_train, y_train, X_test, y_test