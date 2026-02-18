# %%
from federated_rsf.models import (
    FederatedRandomSurvivalForest,
    LocalRandomSurvivalForest,
)
from federated_rsf.schema import SchemaAligner, SchemaCreator, DatasetSchema
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

# Split Dataset samples up to all clients
X_list, y_list = federate_data(
    X,
    y,
    n_clients,
    drop_feature_percentage=0.33,
    random_state=random_state,
)

# Create global Schema
schema_creator = SchemaCreator(anonymize=False)
local_columns = [DatasetSchema(X_local.columns) for X_local in X_list]
dataset_schemas = schema_creator.fit_transform(local_columns)

# Align local datasets
X_list_aligned = []

for X_local, schema in zip(X_list, dataset_schemas):
    schema_aligner = SchemaAligner()
    X_aligned = schema_aligner.fit_transform(X_local, schema)
    X_list_aligned.append(X_aligned)

# Train local models
local_models: list[LocalRandomSurvivalForest] = []

for X_local, y_local in zip(X_list_aligned, y_list):
    local_model = LocalRandomSurvivalForest(
        random_state=random_state,
    )
    local_model = local_model.fit(X_local, y_local)
    local_models.append(local_model)

# Distribute trees between local models
fed_model = FederatedRandomSurvivalForest(local_models=local_models)
fed_model.distribute_trees()


# Example visualization of survival function and cumulative hazard function
client_index = 0
n_lines = 5

survival_local = local_models[client_index].predict_survival_function(
    X_list_aligned[client_index]
)

hazard_local = local_models[client_index].predict_cumulative_hazard_function(
    X_list_aligned[client_index]
)

local_models[client_index].use_federated_estimators()

survival_federated = local_models[client_index].predict_survival_function(
    X_list_aligned[client_index]
)

hazard_federated = local_models[client_index].predict_cumulative_hazard_function(
    X_list_aligned[client_index]
)
from matplotlib import pyplot as plt

for surv in [survival_local, survival_federated]:
    for i, s in enumerate(surv[:n_lines]):
        plt.step(s.x, s.y, where="post", label=str(i))
    plt.ylabel("Survival probability")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.show()


for hazard in [hazard_local, hazard_federated]:
    for i, s in enumerate(hazard[:n_lines]):
        plt.step(s.x, s.y, where="post", label=str(i))
    plt.ylabel("Cumulative hazard")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.show()
