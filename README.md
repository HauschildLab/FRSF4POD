# Federated Random Survival Forest for Partially overlapping Data

federated-rsf is a python implementation of the Federated Random Survival Forest algorithm for partially overlapping data.

## Overview

Federated Random Survival Forest (federated-rsf) enables training
random survival forest models across multiple institutions without sharing raw data.
It is designed for partially overlapping feature spaces and privacy-sensitive
biomedical datasets.

## Features

- Federated survival random forest training
- Support for partially overlapping feature spaces
- Compatible with scikit-survival data structures and evaluation methods
- Privacy-preserving model aggregation


## Installation

### Dependencies

federated-rsf requires:
- numpy (>=2.0.0)
- pandas (>=2.3.0)
- scikit-learn(>=1.8.0)
- scikit-survival (>=0.27.0)

### User installation

The easiest way to install federated-rsf is using pip
```
pip install -U federated-rsf
```

To install in editable mode, clone the repository and then install it using pip
```bash
git https://github.com/HauschildLab/FRSF4POD.git
cd FRSF4POD
pip install -U .
```

To install in editable mode it with optional testing or development libraries uses
```bash
pip install -U -e .[dev]
```
or
```bash
pip install -U -e .[test]
```

## Quick Start

federated-rsf uses three main steps to train federated models.
- First the local data schema of all the clients has to be unified into a global schema
to facilitate the aggregation of models.
- Second is the training of the local-rsf models on the local data
- Third is the aggregation and distribution of the local estimators from the clients.


```python
from federated_rsf.models import (
    FederatedRandomSurvivalForest,
    LocalRandomSurvivalForest,
)
from federated_rsf.preprocessing import SchemaAligner, SchemaCreator
from federated_rsf.testing import create_dummy_data, federate_data

```

In this example we create a dummy dataset using the testing module.
This module can be used valiate the federated learning pipeline is case of missing access to the actual data.

```python
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
```

Next the columns of the local datasets are aligned to a global schema using the `SchemaCreator` and the local
```python
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
```

The local models can then be trained on the processed local data.

```python
# Train local models
local_models: list[LocalRandomSurvivalForest] = []

for X_local, y_local in zip(X_list_aligned, y_list):
    local_model = LocalRandomSurvivalForest(
        random_state=random_state,
    )
    local_model = local_model.fit(X_local, y_local)
    local_models.append(local_model)
```

The trained local models are then aggregated and the estimators are redistributed using the federated model.

```python
# Distribute trees between local models
fed_model = FederatedRandomSurvivalForest(local_models=local_models)
fed_model.distribute_trees()
```

Lastly you can compare the local and the federated model performance for example using the predict `predict_survival_function` and `predict_cumulative_hazard_function`


```python
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
```