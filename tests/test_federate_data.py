# %%
from federated_rsf.testing import federate_data, create_dummy_data
import pytest
import numpy as np


@pytest.mark.parametrize("random_state", [0, 1, 42, None])
def test_random_state(random_state: int):
    n_samples = 1000
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        random_state=0,
    )

    X_list, y_list = federate_data(X, y, clients=3, random_state=random_state)

    X_list_alt, y_list_alt = federate_data(X, y, clients=3, random_state=random_state)

    for X, y, X_alt, y_alt in zip(X_list, y_list, X_list_alt, y_list_alt):
        if random_state is not None:
            np.testing.assert_allclose(X, X_alt)
            np.testing.assert_equal(y["Status"], y_alt["Status"])
            np.testing.assert_allclose(y["Survival_in_days"], y_alt["Survival_in_days"])
        else:
            assert not np.allclose(X, X_alt)
            assert not np.all(y["Status"] == y_alt["Status"])
            assert not np.allclose(y["Survival_in_days"], y_alt["Survival_in_days"])


@pytest.mark.parametrize("drop_percentage", [0, 0.3, 0.5, 0.9, 1])
def test_drop_percentage(drop_percentage):
    random_state = 0
    n_samples = 1000
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        random_state=random_state,
    )

    X_list, _ = federate_data(
        X,
        y,
        clients=3,
        drop_feature_percentage=drop_percentage,
        random_state=random_state,
    )

    for X_fed in X_list:
        assert len(X_fed.columns) == round(len(X.columns) * (1 - drop_percentage))


@pytest.mark.parametrize("clients", [1, 2, 4, 8, 16])
def test_clients_int(clients):
    random_state = 0
    n_samples = 1000
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        random_state=random_state,
    )

    X_list, y_list = federate_data(
        X,
        y,
        clients=clients,
        random_state=random_state,
    )

    assert len(X_list) == len(y_list) == clients
    assert (
        sum(len(X_fed) for X_fed in X_list)
        == sum(len(y_fed) for y_fed in y_list)
        == len(X)
    )

    for X_fed, y_fed in zip(X_list[:-1], y_list[:-1]):
        assert len(X_fed) == len(y_fed) == len(X) // clients


@pytest.mark.parametrize(
    "clients", [[1000], [800, 200], [100, 100, 800], [5, 995], [5, 5, 5, 5, 980]]
)
def test_clients_list(clients):
    random_state = 0
    n_samples = sum(clients)
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        random_state=random_state,
    )

    X_list, y_list = federate_data(
        X,
        y,
        clients=clients,
        random_state=random_state,
    )

    assert len(X_list) == len(y_list) == len(clients)
    assert (
        sum(len(X_fed) for X_fed in X_list)
        == sum(len(y_fed) for y_fed in y_list)
        == len(X)
    )

    for X_fed, y_fed, clients_fed in zip(X_list, y_list, clients):
        assert len(X_fed) == len(y_fed) == clients_fed


def test_shuffle_sync():
    random_state = 0
    n_samples = 1000
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        random_state=random_state,
    )
    X[0] = np.arange(len(X))
    y["Survival_in_days"] = np.arange(len(y))

    X_list, y_list = federate_data(
        X,
        y,
        clients=3,
        random_state=random_state,
    )

    for X_fed, y_fed in zip(X_list, y_list):
        np.testing.assert_allclose(X_fed[0], y_fed["Survival_in_days"])
