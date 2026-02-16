# %%
from federated_rsf.preprocessing import SchemaCreator, SchemaAligner
from federated_rsf.testing import create_dummy_data, federate_data
import pytest


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_alignment(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)

    X_list, _ = federate_data(
        X, y, 5, drop_feature_percentage=0.3, random_state=random_state
    )

    schema, column_maps = SchemaCreator().fit_transform(
        [X_fed.columns for X_fed in X_list]
    )

    X_aligned_list = []

    for X_fed in X_list:
        X_aligned = SchemaAligner().fit_transform(X_fed, schema)
        X_aligned_list.append(X_aligned)

    for X_aligned in X_aligned_list[1:]:
        assert (X_aligned.columns == X_aligned_list[0].columns).all()


@pytest.mark.parametrize("anonymize", [True, False])
def test_add_client(anonymize):
    creator = SchemaCreator(anonymize=anonymize, extra_columns=5, random_state=0)

    local_features = [
        ["age", "gender", "blood_pressure"],
        ["age", "cholesterol", "smoking_status"],
        ["gender", "cholesterol", "exercise_frequency"],
    ]

    creator.fit_transform(local_features)

    creator.add_client(["age", "0"])
    creator.add_client(["1", "2", "3"])
    creator.add_client(["2", "3", "4"])
    with pytest.raises(ValueError):
        creator.add_client(["4", "5"])


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
        ["age", "gender", "blood_pressure"],
        ["age", "cholesterol", "smoking_status"],
        ["gender", "cholesterol", "exercise_frequency"],
    ]

    schema1, column_maps1 = creator1.fit_transform(local_features)
    schema2, column_maps2 = creator2.fit_transform(local_features)

    assert schema1 == schema2
    if random_state is not None:
        assert column_maps1 == column_maps2
    elif anonymize:
        assert column_maps1 != column_maps2


# %%
