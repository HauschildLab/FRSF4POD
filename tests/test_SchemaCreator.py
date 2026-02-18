# %%
from federated_rsf.schema import SchemaCreator, SchemaAligner, DatasetSchema
from federated_rsf.testing import create_dummy_data, federate_data
import pytest
import pandas as pd


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_alignment(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)

    X_list, _ = federate_data(
        X, y, 5, drop_feature_percentage=0.3, random_state=random_state
    )

    dataset_schemas = SchemaCreator().fit_transform(
        [DatasetSchema(X_fed.columns.tolist()) for X_fed in X_list]
    )

    X_aligned_list: list[pd.DataFrame] = []

    for X_fed, schema in zip(X_list, dataset_schemas):
        X_aligned = SchemaAligner().fit_transform(X_fed, schema)
        X_aligned_list.append(X_aligned)

    for X_aligned in X_aligned_list[1:]:
        assert (X_aligned.columns == X_aligned_list[0].columns).all()


@pytest.mark.parametrize("anonymize", [True, False])
def test_add_client(anonymize):
    creator = SchemaCreator(anonymize=anonymize, extra_columns=5, random_state=0)

    local_schemas = [
        DatasetSchema(["age", "gender", "blood_pressure"]),
        DatasetSchema(["age", "cholesterol", "smoking_status"]),
        DatasetSchema(["gender", "cholesterol", "exercise_frequency"]),
    ]

    creator.fit_transform(local_schemas)

    creator.add_client(DatasetSchema(["age", "0"]))
    creator.add_client(DatasetSchema(["1", "2", "3"]))
    creator.add_client(DatasetSchema(["2", "3", "4"]))
    with pytest.raises(ValueError):
        creator.add_client(DatasetSchema(["4", "5"]))


@pytest.mark.parametrize("anonymize", [True, False])
@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, None])
def test_random_state(anonymize, random_state):
    creator1 = SchemaCreator(
        anonymize=anonymize,
        extra_columns=5,
        random_state=random_state,
    )
    creator2 = SchemaCreator(
        anonymize=anonymize,
        extra_columns=5,
        random_state=random_state,
    )

    local_features = [
        DatasetSchema(["age", "gender", "blood_pressure"]),
        DatasetSchema(["age", "cholesterol", "smoking_status"]),
        DatasetSchema(["gender", "cholesterol", "exercise_frequency"]),
    ]

    schemas1 = creator1.fit_transform(local_features)
    schemas2 = creator2.fit_transform(local_features)

    for schema1, schema2 in zip(schemas1, schemas2):
        assert schema1.columns == schema2.columns
        if random_state is not None:
            assert schema1.column_map == schema2.column_map
        elif anonymize:
            assert schema1.column_map != schema2.column_map


# %%
