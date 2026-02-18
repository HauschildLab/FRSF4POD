# %%
from federated_rsf.models import (
    LocalRandomSurvivalForest,
    FederatedRandomSurvivalForest,
)
from federated_rsf.testing import create_dummy_data, federate_data
from federated_rsf.schema import SchemaAligner, DatasetSchema
import pytest
import numpy as np


def test_distribute():
    random_state = 0
    n_features = 64
    n_samples = 4
    n_clients = 3
    X, y = create_dummy_data(
        n_samples * n_clients,
        n_features,
        random_state=random_state,
    )
    X_list, y_list = federate_data(
        X,
        y,
        n_clients,
        random_state=random_state,
    )
    X_list = [SchemaAligner().fit_transform(X_fed, DatasetSchema(X.columns)) for X_fed in X_list]

    local_models = []

    for X_fed, y_fed in zip(X_list, y_list):
        if np.all(y_fed["Status"] == False):
            y_fed["Status"][0] = True
        local_model = LocalRandomSurvivalForest(random_state=random_state)
        local_model = local_model.fit(X_fed, y_fed)
        local_models.append(local_model)

    fed_model = FederatedRandomSurvivalForest(
        random_state=random_state, local_models=local_models
    )

    fed_model.distribute_trees()

    for local_model in local_models:
        assert hasattr(
            local_model, "_federated_estimators"
        ), "_federated_estimators attribute"


def test_save_load(tmp_path):
    random_state = 0
    n_features = 64
    n_samples = 4
    n_clients = 3
    X, y = create_dummy_data(
        n_samples * n_clients,
        n_features,
        random_state=random_state,
    )
    X_list, y_list = federate_data(
        X,
        y,
        n_clients,
        random_state=random_state,
    )
    X_list = [SchemaAligner().fit_transform(X_fed, DatasetSchema(X.columns)) for X_fed in X_list]

    local_models = []

    for X_fed, y_fed in zip(X_list, y_list):
        if np.all(y_fed["Status"] == False):
            y_fed["Status"][0] = True
        local_model = LocalRandomSurvivalForest(random_state=random_state)
        local_model = local_model.fit(X_fed, y_fed)
        local_models.append(local_model)

    fed_model = FederatedRandomSurvivalForest(
        random_state=random_state, local_models=local_models
    )

    path = tmp_path / "model.model"
    fed_model.save(path)
    loaded_model = FederatedRandomSurvivalForest.load(path)

    assert len(fed_model.estimators_) == len(loaded_model.estimators_), "n_estimators"
    assert fed_model.all_features == loaded_model.all_features, "all_features"
    assert len(fed_model.local_models) == len(loaded_model.local_models), "local_models"
    assert all(
        f == f2 for f, f2 in zip(fed_model.tree_features, loaded_model.tree_features)
    ), "tree_features"

    def test_no_fit_predict():
        fed_model = FederatedRandomSurvivalForest(local_models=local_models)
        with pytest.raises(NotImplementedError):
            fed_model.fit(X, y)
        with pytest.raises(NotImplementedError):
            fed_model.predict(X)
