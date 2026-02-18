# %%
from federated_rsf.testing import create_dummy_data
import pytest
import numpy as np


@pytest.mark.parametrize("random_state", [0, 1, 42, None])
def test_random_state(random_state: int):
    n_samples = 1000
    n_features = 10

    X, y = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    X_alt, y_alt = create_dummy_data(
        n_samples,
        n_features,
        drop_feature_percentage=0.33,
        random_state=random_state,
    )

    if random_state is not None:
        np.testing.assert_allclose(X, X_alt)
        np.testing.assert_equal(y["Status"], y_alt["Status"])
        np.testing.assert_allclose(y["Survival_in_days"], y_alt["Survival_in_days"])
    else:
        assert not np.allclose(X, X_alt)
        assert not np.all(y["Status"] == y_alt["Status"])
        assert not np.allclose(y["Survival_in_days"], y_alt["Survival_in_days"])
