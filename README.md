# Federated Survival Random Forest for Partially overlapping Data

federated-rsf is a python implementation of the Federated Survival Random Forest.

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

To install in editable mode first clone this repository and then install it using pip
```
git https://github.com/HauschildLab/FRSF4POD.git
cd FRSF4POD
pip install -U .
```

To install it with optional testing or development libraries uses
```
pip install -U .[dev]
```
or
```
pip install -U .[test]
```


