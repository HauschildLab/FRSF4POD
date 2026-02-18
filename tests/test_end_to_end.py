from federated_rsf.models import (
    FederatedRandomSurvivalForest,
    LocalRandomSurvivalForest,
)
from federated_rsf.testing import create_dummy_data, federate_data
from federated_rsf.schema import SchemaAligner, SchemaCreator, DatasetSchema
import pytest
import numpy as np


@pytest.mark.parametrize("n_samples_per_client", [2, 16])
@pytest.mark.parametrize("n_features", [2, 4])
@pytest.mark.parametrize("update_method", ["all", "constant"])
@pytest.mark.parametrize("update_weighting", ["equal", "site_size"])
def test_end_to_end(n_samples_per_client, n_features, update_method, update_weighting):
    random_state = 0
    n_clients = 4
    n_samples = n_samples_per_client * n_clients

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    X_list, y_list = federate_data(
        X,
        y,
        n_clients,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    dataset_schemas = SchemaCreator(anonymize=True).fit_transform(
        [DatasetSchema(X_fed.columns) for X_fed in X_list]
    )
    X_list = [
        SchemaAligner().fit_transform(X_fed, schema)
        for X_fed, schema in zip(X_list, dataset_schemas)
    ]

    local_models: list[LocalRandomSurvivalForest] = []
    local_predictions = []

    for X_fed, y_fed in zip(X_list, y_list):
        if np.all(y_fed["Status"] == False):
            y_fed["Status"][0] = True
        local_model = LocalRandomSurvivalForest(
            random_state=random_state,
            update_method=update_method,
            update_weighting=update_weighting,
        )
        local_model = local_model.fit(X_fed, y_fed)
        local_predictions.append(local_model.predict(X_fed))
        local_models.append(local_model)

    fed_model = FederatedRandomSurvivalForest(
        random_state=random_state, local_models=local_models
    )

    fed_model.distribute_trees()
    fed_predictions = []

    for local_model, X_fed in zip(local_models, X_list):
        local_model.use_federated_estimators()
        local_model.use_federated_estimators()
        fed_predictions.append(local_model.predict(X_fed))

    local_predictions_2 = []

    for local_model, X_fed in zip(local_models, X_list):
        local_model.use_local_estimators()
        local_model.use_local_estimators()
        local_predictions_2.append(local_model.predict(X_fed))
