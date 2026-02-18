# %%
from federated_rsf.schema import SchemaAligner, DatasetSchema
from federated_rsf.testing import create_dummy_data, federate_data
import pytest


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_columns_equal(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)
    all_columns = X.columns.tolist()

    X_list, _ = federate_data(X, y, 5, random_state=random_state)

    for X_fed in X_list:
        X_aligned = SchemaAligner().fit_transform(X_fed, DatasetSchema(all_columns))

        assert (X_aligned.columns == X.columns).all()


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_columns_map(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)
    all_columns = X.columns.tolist()

    X_list, _ = federate_data(
        X, y, 5, drop_feature_percentage=0.3, random_state=random_state
    )

    for X_fed in X_list:
        renamer = {col: f"renamed_{col}" for col in X_fed.columns}
        renamer_reverse = {val: key for key, val in renamer.items()}
        X_fed = X_fed.rename(columns=renamer)

        assert all([col not in X.columns for col in X_fed.columns])

        X_aligned = SchemaAligner().fit_transform(
            X_fed, DatasetSchema(all_columns, column_map=renamer_reverse)
        )

        assert (X_aligned.columns == X.columns).all()


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_inverse_transform(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)
    all_columns = X.columns.tolist()

    X_list, _ = federate_data(
        X, y, 5, drop_feature_percentage=0.3, random_state=random_state
    )

    for X_fed in X_list:
        renamer = {col: f"renamed_{col}" for col in X_fed.columns}
        renamer_reverse = {val: key for key, val in renamer.items()}
        X_fed = X_fed.rename(columns=renamer)

        schema_aligner = SchemaAligner()

        X_aligned = schema_aligner.fit_transform(
            X_fed, DatasetSchema(all_columns, column_map=renamer_reverse)
        )

        X_undo_transform = schema_aligner.inverse_transform(X_aligned)

        assert (X_undo_transform.columns == X_fed.columns).all()
