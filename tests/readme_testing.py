# %%
from federated_rsf.models import (
    FederatedRandomSurvivalForest,
    LocalRandomSurvivalForest,
)
from federated_rsf.preprocessing import SchemaAligner, SchemaCreator
from federated_rsf.testing import create_dummy_data, federate_data


# Parameters
n_samples = 500
n_features = 10
n_clients = 5
random_state = 0

# Create Dummy Dataset
X, y = create_dummy_data(
    n_samples,
    n_features,
    random_state=random_state,
)

X_list, y_list = federate_data(
    X,
    y,
    n_clients,
    drop_feature_percentage=0.33,
    random_state=random_state,
)

# create global Schema

schema_creator = SchemaCreator(anonymize=False)
local_columns = [X_local.columns for X_local in X_list]

schema, column_maps = schema_creator.fit_transform(local_columns)

# align local datasets
X_list_aligned = []

for X_local, column_map in zip(X_list, column_maps):
    schema_aligner = SchemaAligner()
    X_aligned = schema_aligner.fit_transform(X_local, schema, column_map=column_map)
    X_list_aligned.append(X_aligned)


local_models: list[LocalRandomSurvivalForest] = []

for X_local, y_local in zip(X_list_aligned, y_list):
    local_model = LocalRandomSurvivalForest(
        random_state=random_state,
    )
    local_model = local_model.fit(X_local, y_local)
    local_models.append(local_model)

fed_model = FederatedRandomSurvivalForest(local_models=local_models)
fed_model.distribute_trees()

client_index = 0

res1 = local_models[client_index]._predict_survival_function_parent(
    X_list_aligned[client_index]
)

res2 = local_models[client_index].predict_survival_function(
    X_list_aligned[client_index]
)

has1 = local_models[client_index].predict_cumulative_hazard_function(
    X_list_aligned[client_index]
)

# %%

local_models[client_index].use_federated_estimators()

res3 = local_models[client_index].predict_survival_function(
    X_list_aligned[client_index]
)

has2 = local_models[client_index].predict_cumulative_hazard_function(
    X_list_aligned[client_index]
)
# %%
from matplotlib import pyplot as plt

for res in [res1, res2, res3]:
    for i, s in enumerate(res[:5]):
        plt.step(s.x, s.y, where="post", label=str(i))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

for res in [has1, has2]:
    for i, s in enumerate(res[:5]):
        plt.step(s.x, s.y, where="post", label=str(i))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
