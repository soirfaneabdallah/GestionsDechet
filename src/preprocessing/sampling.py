"""Opérations d'échantillonnage et de fusion sur des jeux (X, y) NumPy."""
import numpy as np


def concat_datasets(x_train, y_train, x_test, y_test,
                    x_train1, y_train1, x_test1, y_test1):
    """
    Concatène deux jeux de données (train/test).
    Les arrays doivent avoir des dimensions compatibles.
    """
    # Concat train
    X_train_full = np.concatenate([x_train, x_train1], axis=0)
    y_train_full = np.concatenate([y_train, y_train1], axis=0)

    # Concat test
    X_test_full = np.concatenate([x_test, x_test1], axis=0)
    y_test_full = np.concatenate([y_test, y_test1], axis=0)

    return X_train_full, y_train_full, X_test_full, y_test_full


def select_data(X_train, y_train, n, shuffle=True, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y_train)
    X_selected, y_selected = [], []

    for cls in classes:
        idx = np.where(y_train == cls)[0]
        if len(idx) < n:
            raise ValueError(f"Classe {cls} n'a que {len(idx)} échantillons, inférieur à n={n}.")
        chosen_idx = np.random.choice(idx, n, replace=False)
        X_selected.append(X_train[chosen_idx])
        y_selected.append(y_train[chosen_idx])

    X_selected = np.concatenate(X_selected, axis=0)
    y_selected = np.concatenate(y_selected, axis=0)

    if shuffle:
        perm = np.random.permutation(len(y_selected))
        X_selected, y_selected = X_selected[perm], y_selected[perm]

    return X_selected, y_selected
