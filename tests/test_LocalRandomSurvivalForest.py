# %%
from federated_rsf.models import LocalRandomSurvivalForest
from federated_rsf.testing import create_dummy_data
import numpy as np
import pytest


@pytest.mark.parametrize("n_samples", [2, 16, 32, 64])
@pytest.mark.parametrize("n_features", [2, 4, 8])
def test_training(n_samples, n_features):
    random_state = 0

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )
    dropped_features = set(X.columns[X.loc[0].isna()])

    model = LocalRandomSurvivalForest(random_state=random_state)

    model.fit(X, y)
    model.predict(X)

    assert model.site_size == len(X), "site size"
    assert model.local_features == (
        set(range(n_features)) - dropped_features
    ), "local_features"
    for estimator in model.estimators_:
        assert estimator.sample_weight == len(X) / 100, "estimator weight"


def test_save_load(tmp_path):
    random_state = 0
    n_samples, n_features = 128, 32

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    model = LocalRandomSurvivalForest(random_state=random_state)
    model.fit(X, y)

    path = tmp_path / "model.model"
    model.save(path)
    loaded_model = LocalRandomSurvivalForest.load(path)

    np.testing.assert_array_equal(
        model.predict(X),
        loaded_model.predict(X),
        err_msg="predict",
    )
    np.testing.assert_array_equal(
        model.predict_survival_function(X),
        loaded_model.predict_survival_function(X),
        err_msg="predict_survival_function",
    )

    assert model.site_size == loaded_model.site_size, "site_size"
    assert model.local_features == loaded_model.local_features, "local_features"
    for estimator, loaded_estimator in zip(model.estimators_, loaded_model.estimators_):
        assert (
            estimator.sample_weight == loaded_estimator.sample_weight
        ), "sample_weight"


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, None])
def test_local_fed_switch(random_state):
    n_samples, n_features = 128, 4

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    model = LocalRandomSurvivalForest(random_state=random_state)
    model.fit(X, y)

    pred_local = model.predict(X)

    model.set_federated_estimators(
        np.random.default_rng(random_state).choice(model.estimators_, size=80)
    )

    model.use_federated_estimators()
    assert (
        model.tree_origin == "federated"
    ), "model_status after use_federated_estimators"
    pred_fed = model.predict(X)

    model.use_local_estimators()
    assert model.tree_origin == "local", "model_status after use_local_estimators"
    pred_local_2 = model.predict(X)
    assert np.allclose(pred_local, pred_local_2), "predictions should be equal"

    model.use_federated_estimators()
    pred_fed_2 = model.predict(X)
    assert np.allclose(pred_fed, pred_fed_2), "predictions should be equal"

    assert not np.allclose(
        pred_local, pred_fed
    ), "local and federated predictions should differ"


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, None])
def test_hazard_survival_function(random_state):
    n_samples, n_features = 128, 4

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    model = LocalRandomSurvivalForest(random_state=random_state)

    model.fit(X, y)

    hazard_our = model.predict_cumulative_hazard_function(X)
    hazard_original = model._predict_cumulative_hazard_function_parent(X)

    survival_our = model.predict_survival_function(X)
    survival_original = model._predict_survival_function_parent(X)

    assert np.all(hazard_our == hazard_original)
    assert np.all(survival_our == survival_original)
