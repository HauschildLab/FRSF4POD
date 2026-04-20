#!/usr/bin/env python
# coding: utf-8

# # Federated RSF — Per-Client Convergence Benchmark (BRCA)
# 
# For each of 4 client test sets, we predict with 4 models:
# 
# | Model | Description |
# |-------|-------------|
# | **Local** | Trained only on that client's data |
# | **Fed (2 clients)** | Federated between clients 1–2 |
# | **Fed (3 clients)** | Federated between clients 1–3 |
# | **Fed (4 clients)** | Federated across all 4 clients |
# 
# **Design:** 4 clients, non-overlapping patient cohorts and feature subsets.
# Each client has its own train/test split. There is **no global test set**.
# For each test set *i* and federation level *k*:
#   - *i* ≤ *k*: client *i*'s model received cross-client trees → federated.
#   - *i* > *k*: client *i* has not joined yet → local model result (no change).
# 
# **Metrics:** C-index, RMST Δ, Risk at Time, Kaplan–Meier curves.

# In[34]:


import warnings
warnings.filterwarnings('ignore')

import copy
import sys
from pathlib import Path
from itertools import combinations as _combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.interpolate import interp1d

from sksurv.datasets import load_breast_cancer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

try:
    from federated_rsf.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
    from federated_rsf.schema import DatasetSchema, SchemaAligner, SchemaCreator
    from federated_rsf.testing import federate_data
except ModuleNotFoundError as exc:
    # Re-raise if the missing import is not our package.
    if exc.name != 'federated_rsf':
        raise
    # Allow running this example directly without installing the package.
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / 'src'
    for p in (src_root, project_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    from federated_rsf.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
    from federated_rsf.schema import DatasetSchema, SchemaAligner, SchemaCreator
    from federated_rsf.testing import federate_data

# ── Publication-grade plot style ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 120,
    'savefig.dpi'       : 300,
    'font.size'         : 11,
    'font.family'       : 'DejaVu Sans',
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 11,
    'axes.labelweight'  : 'bold',
    'axes.facecolor'    : 'white',
    'axes.edgecolor'    : '#222222',
    'axes.linewidth'    : 1.0,
    'axes.grid'         : True,
    'grid.color'        : '#dddddd',
    'grid.linewidth'    : 0.7,
    'grid.alpha'        : 1.0,
    'figure.facecolor'  : 'white',
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'legend.fontsize'   : 9,
    'legend.framealpha' : 0.95,
    'legend.edgecolor'  : '#aaaaaa',
    'lines.linewidth'   : 1.8,
    'patch.linewidth'   : 0.8,
})
print('Imports OK')


# ## 1. Configuration

# In[35]:


N_CLIENTS           = 4      # hospital sites
DROP_FEATURE_PCT    = 0.35   # fraction of features dropped per client
N_FOLDS             = 5      # stratified CV folds per client
N_ESTIMATORS        = 300    # trees per local model
RANDOM_STATE        = 999
UPDATE_METHOD       = 'constant'
MIN_FEATURE_OVERLAP = 1.0    # 1.0 = strict subset (safe).  <1.0 admits
                             # cross-client trees whose features only partly
                             # overlap with the recipient — see notes in
                             # FederatedRandomSurvivalForest.distribute_trees.

K_LABELS = {
    1: 'Local',
    2: 'Fed (2 clients)',
    3: 'Fed (3 clients)',
    4: 'Fed (4 clients)',
}
K_ORDERED = [K_LABELS[k] for k in range(1, N_CLIENTS + 1)]

COLORS = {
    'Local'            : '#111111',
    'Fed (2 clients)'  : '#444444',
    'Fed (3 clients)'  : '#777777',
    'Fed (4 clients)'  : '#aaaaaa',
    'Centralized RSF'  : '#005599',   # blue — trained on all pooled data
}
HATCHES = {
    'Local'            : '',
    'Fed (2 clients)'  : '///',
    'Fed (3 clients)'  : '...',
    'Fed (4 clients)'  : 'xxx',
    'Centralized RSF'  : '---',
}
LINESTYLES = {
    'Local'            : '-',
    'Fed (2 clients)'  : '--',
    'Fed (3 clients)'  : '-.',
    'Fed (4 clients)'  : ':',
    'Centralized RSF'  : (0, (3, 1, 1, 1)),   # dash-dot-dot
}

CLIENT_COLORS     = ['#000000', '#333333', '#666666', '#999999'][:N_CLIENTS]
CLIENT_MARKERS    = ['o', 's', '^', 'D'][:N_CLIENTS]
CLIENT_LINESTYLES = ['-', '--', '-.', ':'][:N_CLIENTS]
C_HIGH = '#CC2222'
C_LOW  = '#2266CC'

print(f'Clients: {N_CLIENTS} | Drop: {DROP_FEATURE_PCT*100:.0f}% | '
      f'CV folds: {N_FOLDS} | Trees: {N_ESTIMATORS}')


# ## 2. Load Dataset
# 
# **Breast Cancer Gene Expression** dataset (sksurv): 198 patients, 82 features, time in days.
# Adaptive time horizons are derived from event-time percentiles.

# In[36]:


X_raw, y_raw = load_breast_cancer()
X_full = OneHotEncoder().fit_transform(X_raw)

event_field = y_raw.dtype.names[0]   # 'e.tdm'
time_field  = y_raw.dtype.names[1]   # 't.tdm'

all_times   = y_raw[time_field]
event_times = all_times[y_raw[event_field]]

T_SHORT      = np.percentile(event_times, 25)
T_MED        = np.percentile(event_times, 50)
T_LONG       = np.percentile(event_times, 75)
RMST_HORIZON = np.percentile(event_times, 80)

print(f'Samples : {len(X_full)}   Features: {X_full.shape[1]}')
print(f'Events  : {y_raw[event_field].sum()} ({y_raw[event_field].mean()*100:.1f}%)')
print(f'Follow-up: {all_times.min():.0f}–{all_times.max():.0f} d  '
      f'(median {np.median(all_times):.0f} d = {np.median(all_times)/365:.1f} yr)')
print(f'\nAdaptive time horizons (days):')
print(f'  T_short = {T_SHORT:.0f}  ({T_SHORT/365:.1f} yr)')
print(f'  T_med   = {T_MED:.0f}  ({T_MED/365:.1f} yr)')
print(f'  T_long  = {T_LONG:.0f}  ({T_LONG/365:.1f} yr)')
print(f'  RMST    = {RMST_HORIZON:.0f}  ({RMST_HORIZON/365:.1f} yr)')


# ## 3. Per-Client Data
#
# Data flow:
#   `X_full` → `federate_data` → `X_clients_raw[i]`
#       (non-overlapping patient cohorts, each with 65 % of features)
#   `X_clients_raw[i]` → `StratifiedKFold(N_FOLDS)` → per-fold train/test indices
#       (per-client, per-fold; `StandardScaler` fitted on fold-train only)
#
# There is **no global test set**. All evaluation is on per-client test sets,
# repeated across N_FOLDS cross-validation folds.

# In[37]:


X_clients_raw, y_clients = federate_data(
    X_full, y_raw,
    clients=N_CLIENTS,
    drop_feature_percentage=DROP_FEATURE_PCT,
    random_state=RANDOM_STATE,
)
client_feat_sets = [sorted(X_c.columns.tolist()) for X_c in X_clients_raw]

# Schema alignment — required by FederatedRSF.distribute_trees()
schema_list       = [DatasetSchema(cols) for cols in client_feat_sets]
schema_creator    = SchemaCreator(anonymize=False)
federated_schemas = schema_creator.fit_transform(schema_list)
global_columns    = federated_schemas[0].columns

X_clients_aligned = [SchemaAligner().fit_transform(X_c, s)
                     for X_c, s in zip(X_clients_raw, federated_schemas)]

missing = sorted(set(X_full.columns) - set(global_columns))
print(f'Original features : {X_full.shape[1]}')
print(f'Federated union   : {len(global_columns)}  '
      f'(absent from all clients: {missing if missing else "none"})')

# Precompute per-client 5-fold CV splits (StratifiedKFold on event indicator,
# fall back to KFold if any class has <N_FOLDS samples).
fold_splits = []  # fold_splits[i] = list of (train_idx, test_idx) tuples, length N_FOLDS
for i in range(N_CLIENTS):
    y_c = y_clients[i]
    events_c = y_c[event_field].astype(int)
    n_pos = int(events_c.sum())
    n_neg = int(len(events_c) - n_pos)
    if min(n_pos, n_neg) >= N_FOLDS:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits_i = list(splitter.split(X_clients_raw[i], events_c))
    else:
        print(f'  C{i+1}: min class count ({min(n_pos, n_neg)}) < {N_FOLDS}; '
              f'falling back to non-stratified KFold')
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits_i = list(splitter.split(X_clients_raw[i]))
    fold_splits.append(splits_i)

print(f'\nPrecomputed {N_FOLDS}-fold CV splits for each of {N_CLIENTS} clients.')
print(f"{'Client':<10} {'N':>6} {'Features':>10} {'Events':>8} {'Event rate':>12} "
      f"{'Median t':>10}")
print('─' * 64)
_event_rates = []
_median_times = []
for i in range(N_CLIENTS):
    n_i  = len(X_clients_raw[i])
    ev_i = int(y_clients[i][event_field].sum())
    er_i = y_clients[i][event_field].mean()
    mt_i = float(np.median(y_clients[i][time_field]))
    _event_rates.append(er_i)
    _median_times.append(mt_i)
    print(f"  C{i+1:<7} {n_i:>6} {len(client_feat_sets[i]):>10} "
          f"{ev_i:>8} {er_i*100:>10.1f}%  {mt_i:>10.0f}")
# Non-IID-ness indicator: spread in event rate and median time across clients.
print(f'\nHeterogeneity across clients:')
print(f'  Event-rate range : {min(_event_rates)*100:.1f}% – {max(_event_rates)*100:.1f}% '
      f'(spread {(max(_event_rates) - min(_event_rates))*100:.1f} pp)')
print(f'  Median-time range: {min(_median_times):.0f} – {max(_median_times):.0f} d '
      f'(spread {max(_median_times) - min(_median_times):.0f} d)')
print('  (Spreads >10 pp event rate or >365 d median → meaningfully non-IID; '
      'cross-client trees encode cohort-specific signals that may not transfer.)')


# ## 4. Helper Functions

# In[38]:


def c_idx(risk, y):
    return concordance_index_censored(y[event_field], y[time_field], risk)[0]


def compute_rmst(events, times, horizon):
    """Trapezoidal RMST: area under KM curve from 0 to horizon."""
    try:
        km_t, km_s = kaplan_meier_estimator(events, times)
        mask = km_t <= horizon
        if not mask.any():
            return float(horizon)
        t = np.concatenate([[0.0], km_t[mask], [horizon]])
        s = np.concatenate([[1.0], km_s[mask], [km_s[mask][-1]]])
        return float(np.trapezoid(s, t))
    except Exception:
        return np.nan


def compute_risk_at_t(events, times, t_query):
    """Cumulative incidence at t_query: 1 − S(t)."""
    try:
        km_t, km_s = kaplan_meier_estimator(events, times)
        idx  = np.searchsorted(km_t, t_query, side='right') - 1
        surv = km_s[max(0, idx)] if idx >= 0 else 1.0
        return float(1.0 - surv)
    except Exception:
        return np.nan


def km_greenwood_ci(events, times, alpha=0.05):
    """(t, S, lower_CI, upper_CI) via Greenwood's variance + log-log transform."""
    km_t, km_s = kaplan_meier_estimator(events, times)
    t_sorted = np.sort(times)
    e_sorted = events[np.argsort(times)]
    z = stats.norm.ppf(1 - alpha / 2)
    var_cumsum = 0.0
    lower, upper = [], []
    for j, t_j in enumerate(km_t):
        at_risk     = np.sum(t_sorted >= t_j)
        events_at_t = np.sum((t_sorted == t_j) & e_sorted)
        if at_risk == 0 or events_at_t == 0:
            lower.append(np.nan); upper.append(np.nan); continue
        var_cumsum += events_at_t / (at_risk * (at_risk - events_at_t + 1e-9))
        s = float(km_s[j])
        if s <= 0 or s >= 1:
            lower.append(np.nan); upper.append(np.nan); continue
        log_log_s  = np.log(-np.log(s))
        se_log_log = np.sqrt(var_cumsum) / np.abs(np.log(s))
        lower.append(np.clip(np.exp(-np.exp(log_log_s + z * se_log_log)), 0, 1))
        upper.append(np.clip(np.exp(-np.exp(log_log_s - z * se_log_log)), 0, 1))
    return km_t, km_s, np.array(lower), np.array(upper)


def _fed_combined_predict(model, X, n_local=N_ESTIMATORS):
    """
    'add' semantics: tree-count-weighted average of local + cross-client predictions.
    Falls back to local-only prediction when no cross-client trees are present.
    """
    n_cross = len(model._federated_estimators)
    model.use_local_estimators()
    r_local = model.predict(X)
    if n_cross == 0:
        return r_local
    model.use_federated_estimators()
    r_cross = model.predict(X)
    model.use_local_estimators()
    return (n_local * r_local + n_cross * r_cross) / (n_local + n_cross)


def risk_group_metrics(risk_scores, y):
    """RMST and Risk@T for high/low risk groups (median split)."""
    events = y[event_field]
    times  = y[time_field]
    high   = risk_scores >= np.median(risk_scores)
    low    = ~high
    out    = {}
    for grp, mask in [('high', high), ('low', low)]:
        e, t = events[mask], times[mask]
        out[f'rmst_{grp}']       = compute_rmst(e, t, RMST_HORIZON)
        out[f'risk_short_{grp}'] = compute_risk_at_t(e, t, T_SHORT)
        out[f'risk_med_{grp}']   = compute_risk_at_t(e, t, T_MED)
        out[f'risk_long_{grp}']  = compute_risk_at_t(e, t, T_LONG)
    out['rmst_diff']       = out['rmst_low']        - out['rmst_high']
    out['risk_short_diff'] = out['risk_short_high']  - out['risk_short_low']
    out['risk_med_diff']   = out['risk_med_high']    - out['risk_med_low']
    out['risk_long_diff']  = out['risk_long_high']   - out['risk_long_low']
    return out


# ## 5. 5-Fold Cross-Validation Loop
#
# For each fold f ∈ {0..N_FOLDS-1}:
#   1. Build per-client fold-train / fold-test with per-client Z-score normalisation.
#   2. Train local models (one per client).
#   3. Train centralized baseline on pooled fold-train data.
#   4. Cumulative federation k=1..N_CLIENTS + per-(k,i) C-index, risk.
#   5. All-subset enumeration for convergence plot.

# In[39]:


# Per-fold containers. All keys are fold index f ∈ [0, N_FOLDS).
fold_X_trains  = [None] * N_FOLDS   # fold_X_trains[f][i]
fold_X_tests   = [None] * N_FOLDS
fold_y_trains  = [None] * N_FOLDS
fold_y_tests   = [None] * N_FOLDS
fold_local_models = [None] * N_FOLDS
fold_cent_c    = [None] * N_FOLDS   # fold_cent_c[f][i]
fold_cent_risk = [None] * N_FOLDS
fold_eval      = [None] * N_FOLDS   # fold_eval[f][k][i] = {'c_index', 'risk', 'n_fed_trees'}
fold_subset_pc = [None] * N_FOLDS   # fold_subset_pc[f][(frozenset, client_i)] = c_index

cent_n_est = N_CLIENTS * N_ESTIMATORS   # same tree budget as the full local ensemble

for f in range(N_FOLDS):
    print(f'\n════════════ Fold {f+1}/{N_FOLDS} ════════════')
    X_trs, X_tes, y_trs, y_tes = [], [], [], []
    for i in range(N_CLIENTS):
        tr_idx, te_idx = fold_splits[i][f]
        X_c = X_clients_raw[i][client_feat_sets[i]]
        y_c = y_clients[i]
        # No per-client StandardScaler: RSF splits are order-based and invariant
        # to monotonic transforms. Per-client scaling would make tree split
        # thresholds learned on client A unusable on client B (different mean/σ
        # per feature), injecting noise into cross-client trees.
        #
        # Align each client's X to the global column layout so that every
        # tree's integer feature indices refer to the same global feature
        # across all clients. Train gets NaN for absent features (so
        # LocalRandomSurvivalForest.fit can still detect `local_features`
        # from the NaN pattern); test gets 0-fill so predict works directly.
        X_tr = (
            X_c.iloc[tr_idx]
                .reset_index(drop=True)
                .reindex(columns=global_columns)            # NaN for absent
        )
        X_te = (
            X_c.iloc[te_idx]
                .reset_index(drop=True)
                .reindex(columns=global_columns, fill_value=0.0)
        )
        X_trs.append(X_tr)
        X_tes.append(X_te)
        y_trs.append(y_c[tr_idx])
        y_tes.append(y_c[te_idx])
    fold_X_trains[f] = X_trs
    fold_X_tests[f]  = X_tes
    fold_y_trains[f] = y_trs
    fold_y_tests[f]  = y_tes

    # ── Local models ────────────────────────────────────────────────────────
    local_models_f = []
    for i in range(N_CLIENTS):
        m = LocalRandomSurvivalForest(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE + f,
            update_method=UPDATE_METHOD, min_samples_split=6, min_samples_leaf=3,
        )
        m.fit(X_trs[i], y_trs[i])
        local_models_f.append(m)
    fold_local_models[f] = local_models_f

    # ── Centralized baseline ────────────────────────────────────────────────
    # X_trs are already aligned to global_columns with NaN for absent features;
    # sklearn's RSF can't fit with NaN, so replace with 0 for the pooled model.
    cent_train_parts = [X_trs[i].fillna(0.0) for i in range(N_CLIENTS)]
    X_cent_train = pd.concat(cent_train_parts, ignore_index=True)
    y_cent_train = np.concatenate([y_trs[i] for i in range(N_CLIENTS)])
    centralized_model = RandomSurvivalForest(
        n_estimators=cent_n_est, random_state=RANDOM_STATE + f,
        min_samples_split=6, min_samples_leaf=3,
    )
    centralized_model.fit(X_cent_train, y_cent_train)

    cent_risk_f = {}
    cent_c_f    = {}
    for i in range(N_CLIENTS):
        # X_tes[i] is already aligned to global_columns with 0-fill.
        r = centralized_model.predict(X_tes[i])
        cent_risk_f[i] = r
        cent_c_f[i]   = c_idx(r, y_tes[i])
    fold_cent_c[f]    = cent_c_f
    fold_cent_risk[f] = cent_risk_f

    # ── Cumulative federation k=1..N_CLIENTS ────────────────────────────────
    eval_f = {}
    for k in range(1, N_CLIENTS + 1):
        models_k = [copy.deepcopy(local_models_f[i]) for i in range(N_CLIENTS)]
        if k >= 2:
            fed = FederatedRandomSurvivalForest(local_models=models_k[:k])
            fed.distribute_trees(min_feature_overlap=MIN_FEATURE_OVERLAP)
        eval_f[k] = {}
        for i in range(N_CLIENTS):
            n_cross = len(models_k[i]._federated_estimators)
            risk    = _fed_combined_predict(models_k[i], X_tes[i])
            ci      = c_idx(risk, y_tes[i])
            eval_f[k][i] = {'c_index': ci, 'risk': risk, 'n_fed_trees': n_cross}
    fold_eval[f] = eval_f

    # ── All-subset enumeration (for convergence plot) ───────────────────────
    subset_pc_f = {}
    for k in range(1, N_CLIENTS + 1):
        for subset in _combinations(range(N_CLIENTS), k):
            subset_models = [copy.deepcopy(local_models_f[i]) for i in subset]
            if k >= 2:
                fed_sub = FederatedRandomSurvivalForest(local_models=subset_models)
                fed_sub.distribute_trees(min_feature_overlap=MIN_FEATURE_OVERLAP)
            fs = frozenset(subset)
            for j, i in enumerate(subset):
                risk = _fed_combined_predict(subset_models[j], X_tes[i])
                subset_pc_f[(fs, i)] = c_idx(risk, y_tes[i])
    fold_subset_pc[f] = subset_pc_f

    # Fold summary: Local / Fed(N) / Centralized mean C-index across clients
    loc_mean  = np.mean([eval_f[1][i]['c_index'] for i in range(N_CLIENTS)])
    fed_mean  = np.mean([eval_f[N_CLIENTS][i]['c_index'] for i in range(N_CLIENTS)])
    cent_mean = np.mean(list(cent_c_f.values()))
    print(f'  Fold {f+1}: Local={loc_mean:.4f}  Fed({N_CLIENTS})={fed_mean:.4f}  '
          f'Centralized={cent_mean:.4f}')
    # Cross-client tree admission counts per (k, i) — diagnoses asymmetric
    # filtering from non-overlapping feature subsets (distribute_trees only
    # admits trees whose features are a subset of the recipient's features).
    print(f'    n_fed_trees by k × client (local budget = {N_ESTIMATORS}):')
    for k in range(2, N_CLIENTS + 1):
        counts = [eval_f[k][i]['n_fed_trees'] for i in range(N_CLIENTS)]
        print(f'      k={k}: ' + '  '.join(f'C{i+1}={c}' for i, c in enumerate(counts)))


# ── Fold-0 aliases used by feature-importance section and any fold-specific plots ──
X_client_trains = fold_X_trains[0]
X_client_tests  = fold_X_tests[0]
y_client_trains = fold_y_trains[0]
y_client_tests  = fold_y_tests[0]
local_models    = fold_local_models[0]
cent_c          = fold_cent_c[0]
cent_risk       = fold_cent_risk[0]
eval_results    = fold_eval[0]
subset_pc       = fold_subset_pc[0]


# ## 7. Results Summary Table (fold-averaged)
#
# Each cell shows mean ± SD across the 5 CV folds.

# In[42]:


MODEL_NAMES_ALL = K_ORDERED + ['Centralized RSF']


def _model_folds(name, i):
    """Return length-N_FOLDS array of C-index for (model, client i)."""
    if name == 'Centralized RSF':
        return np.array([fold_cent_c[f][i] for f in range(N_FOLDS)])
    k = list(K_LABELS.keys())[list(K_LABELS.values()).index(name)]
    return np.array([fold_eval[f][k][i]['c_index'] for f in range(N_FOLDS)])


rows = []
for i in range(N_CLIENTS):
    n_te_avg = int(np.mean([len(fold_y_tests[f][i]) for f in range(N_FOLDS)]))
    ev_avg   = np.mean([fold_y_tests[f][i][event_field].sum() for f in range(N_FOLDS)])
    row = {'Test Set': f'C{i+1}  (n≈{n_te_avg}, ev≈{ev_avg:.1f})'}
    for name in MODEL_NAMES_ALL:
        v = _model_folds(name, i)
        row[name] = f'{v.mean():.4f} ± {v.std(ddof=1):.4f}'
    rows.append(row)

# "Mean" row: average across clients of per-fold means
mean_row = {'Test Set': 'Mean'}
for name in MODEL_NAMES_ALL:
    vals = np.concatenate([_model_folds(name, i) for i in range(N_CLIENTS)])
    mean_row[name] = f'{vals.mean():.4f} ± {vals.std(ddof=1):.4f}'
rows.append(mean_row)

summary_df = pd.DataFrame(rows).set_index('Test Set')
print(f'C-index — rows: client test sets, cols: federation level (mean ± SD over {N_FOLDS} folds)')
print('(When client i > k, it has not yet joined federation → same as local model)')
print()
print(summary_df.to_string())

# ── Oracle-ceiling diagnostic ────────────────────────────────────────────────
# Centralized − Local is the maximum C-index gain federation could hope to
# recover. If this gap is small, no amount of algorithmic improvement to the
# federation scheme will show dramatic gains on this dataset.
_local_pooled = np.concatenate([_model_folds('Local', i)           for i in range(N_CLIENTS)])
_fedN_pooled  = np.concatenate([_model_folds(K_LABELS[N_CLIENTS], i) for i in range(N_CLIENTS)])
_cent_pooled  = np.concatenate([_model_folds('Centralized RSF', i) for i in range(N_CLIENTS)])
_gap_ceiling  = _cent_pooled.mean() - _local_pooled.mean()
_gap_realised = _fedN_pooled.mean()  - _local_pooled.mean()
_recovered    = (_gap_realised / _gap_ceiling * 100) if abs(_gap_ceiling) > 1e-6 else float('nan')
print()
print('Oracle ceiling (pooled means):')
print(f'  Local            = {_local_pooled.mean():.4f}')
print(f'  Fed({N_CLIENTS})           = {_fedN_pooled.mean():.4f}  '
      f'(Δ vs Local = {_gap_realised:+.4f})')
print(f'  Centralized RSF  = {_cent_pooled.mean():.4f}  '
      f'(Δ vs Local = {_gap_ceiling:+.4f}  ← oracle ceiling)')
print(f'  Federation recovered {_recovered:+.1f}% of the available Local→Centralized gap.')
print('  (Gap <0.03 → dataset is too small for federation gains to be distinguishable '
      'from CV noise.)')


# ## 7b. Cross-Validation 5–95 % Percentile Intervals
#
# Per-model interval summarises the distribution of per-fold pooled C-index.

# In[43]:


def _pooled_cindex(name, f):
    """Pooled-across-clients C-index for a given (model, fold)."""
    if name == 'Centralized RSF':
        risks = np.concatenate([fold_cent_risk[f][i] for i in range(N_CLIENTS)])
    else:
        k = list(K_LABELS.keys())[list(K_LABELS.values()).index(name)]
        risks = np.concatenate([fold_eval[f][k][i]['risk'] for i in range(N_CLIENTS)])
    y_all = np.concatenate([fold_y_tests[f][i] for i in range(N_CLIENTS)])
    return c_idx(risks, y_all)


print(f'Per-fold pooled C-index summary (mean ± SD, 5–95 pctile across {N_FOLDS} folds):\n')
print(f"{'Model':<22} {'Mean':>8}  {'SD':>7}  {'5–95 % CI':>20}")
print('─' * 60)
ci_table = {}
for name in MODEL_NAMES_ALL:
    pooled = np.array([_pooled_cindex(name, f) for f in range(N_FOLDS)])
    m, s = pooled.mean(), pooled.std(ddof=1)
    lo, hi = np.percentile(pooled, [5, 95])
    ci_table[name] = (m, lo, hi)
    print(f'{name:<22} {m:>8.4f}  {s:>7.4f}  [{lo:.4f}, {hi:.4f}]')


# ## 8. Convergence Plot — All Client Subset Combinations
# 
# For every non-empty subset S of clients:
#   - Federate the models for clients in S.
#   - Evaluate **each client i ∈ S** on its own test set using its model from the federation.
#   - Plot one dot per (subset, client) pair at x = |S|.
# 
# k=1 (local):  4 subsets × 1 client  =  4 dots
# k=2:          6 subsets × 2 clients = 12 dots
# k=3:          4 subsets × 3 clients = 12 dots
# k=4:          1 subset  × 4 clients =  4 dots
# Centralized:  1 model   × 4 test sets = 4 dots
# 
# Dots are colored by **which client's test set** is being evaluated.

# In[44]:


# ── Build arrays for scatter (enumerated during fold loop) ───────────────────
# Each (subset, client_i) point has N_FOLDS C-index values.
subset_entries = []  # list of dicts: {k, subset_label, client_idx, vals (len N_FOLDS)}
for k in range(1, N_CLIENTS + 1):
    for subset in sorted(_combinations(range(N_CLIENTS), k)):
        fs    = frozenset(subset)
        label = '{' + ','.join(f'C{i+1}' for i in sorted(subset)) + '}'
        for i in subset:
            vals = np.array([fold_subset_pc[f][(fs, i)] for f in range(N_FOLDS)])
            subset_entries.append({
                'k': k, 'subset_label': label, 'client_idx': i, 'vals': vals,
            })

n_entries = len(subset_entries)
CENT_X = N_CLIENTS + 0.80

rng_j  = np.random.default_rng(42)
entry_jitter = rng_j.uniform(-0.32, 0.32, n_entries)

# Centralized: 4 dots evenly spaced around CENT_X (one per test set)
cent_jitter = np.linspace(-0.20, 0.20, N_CLIENTS)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# Federated dots (colored by client test set): faded per-fold dots + bold mean
for idx, ent in enumerate(subset_entries):
    cli  = ent['client_idx']
    x0   = ent['k'] + entry_jitter[idx]
    vals = ent['vals']
    # Faded per-fold dots
    ax.scatter([x0] * N_FOLDS, vals,
               color=CLIENT_COLORS[cli], marker=CLIENT_MARKERS[cli],
               s=25, edgecolors='none', alpha=0.25, zorder=4)
    # Bold mean dot
    ax.scatter(x0, vals.mean(),
               color=CLIENT_COLORS[cli], marker=CLIENT_MARKERS[cli],
               s=80, edgecolors='#333333', linewidths=0.6, alpha=0.95, zorder=5)

# Centralized: faded per-fold dots + bold mean per test set
for i in range(N_CLIENTS):
    cent_vals = np.array([fold_cent_c[f][i] for f in range(N_FOLDS)])
    x0 = CENT_X + cent_jitter[i]
    ax.scatter([x0] * N_FOLDS, cent_vals,
               color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=35, edgecolors='none', alpha=0.25, zorder=6)
    ax.scatter(x0, cent_vals.mean(),
               color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=180, edgecolors=COLORS['Centralized RSF'], linewidths=2.0,
               alpha=0.95, zorder=7)
    ax.annotate(f'C{i+1}',
                (x0, cent_vals.mean()),
                xytext=(0, 7), textcoords='offset points',
                ha='center', fontsize=7.5, fontweight='bold',
                color=COLORS['Centralized RSF'])

ax.axhline(0.5, color='#aaaaaa', ls=':', lw=1.0)

# ── Legend ────────────────────────────────────────────────────────────────────
for i in range(N_CLIENTS):
    ax.scatter([], [], color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=80, edgecolors='#333333', linewidths=0.6,
               label=f'C{i+1} test set')
from matplotlib.lines import Line2D as _L2D
ax.add_artist(ax.legend(
    fontsize=9, loc='lower right', ncol=2, title='Test set (color)'))

cent_handle = Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='#888888',
                     markeredgecolor=COLORS['Centralized RSF'],
                     markeredgewidth=2.0, markersize=10,
                     label='Centralized RSF\n(thick outline)')
ax.legend(handles=[cent_handle], fontsize=9, loc='upper right')

# ── X-axis ticks with dot counts ─────────────────────────────────────────────
k_tick_pos = list(range(1, N_CLIENTS + 1)) + [CENT_X]
k_tick_names = []
for k in range(1, N_CLIENTS + 1):
    n_combos = len(list(_combinations(range(N_CLIENTS), k)))
    n_dots   = sum(1 for ent in subset_entries if ent['k'] == k)
    k_tick_names.append(
        f'{K_LABELS[k].replace(" (", chr(10)+"(")}\n'
        f'({n_combos} combo{"s" if n_combos > 1 else ""}, {n_dots} dots)')
k_tick_names.append(f'Centralized\nRSF\n({N_CLIENTS} dots)')

ax.set_xticks(k_tick_pos)
ax.set_xticklabels(k_tick_names, fontsize=8.5)
ax.set_xlabel('Federation size k  ·  dot color = which client test set is evaluated')
ax.set_ylabel('C-index (evaluated on that client\'s own test set)')
ax.set_title(f'Federated Convergence — All Client Subset Combinations\n'
             f'Bold dot = mean across {N_FOLDS} CV folds  ·  faded dots = per-fold values  '
             f'·  thick outline = Centralized RSF',
             fontweight='bold')
plt.tight_layout()
plt.savefig('convergence.pdf', bbox_inches='tight')
plt.show()


# ## 9. C-index Boxplot
#
# Left: per-client boxplots (one box per model, distribution over CV folds).
# Right: pooled-across-clients boxplots (N_FOLDS × N_CLIENTS values per box).

# In[45]:


def _style_box(bp, color, hatch):
    for patch in bp['boxes']:
        patch.set(facecolor=color, edgecolor='#111111', linewidth=0.7, hatch=hatch)
    for median in bp['medians']:
        median.set(color='#111111', linewidth=1.3)
    for whisker in bp['whiskers']:
        whisker.set(color='#111111', linewidth=0.9)
    for cap in bp['caps']:
        cap.set(color='#111111', linewidth=0.9)
    for mean in bp.get('means', []):
        mean.set(color='#d62728', linewidth=1.3, linestyle='-')


def _annotate_mean_std(ax, x, y, mean_val, std_val, mean_fmt, std_fmt,
                       mean_fs=7.0, std_fs=5.8, color='#111111'):
    """Annotate mean and SD with a smaller SD line."""
    if not np.isfinite(mean_val):
        return
    ax.text(x, y, mean_fmt.format(mean_val),
            ha='center', va='bottom', fontsize=mean_fs,
            fontweight='bold', color=color)
    if np.isfinite(std_val):
        ax.annotate(std_fmt.format(std_val), (x, y),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=std_fs, color=color)


fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [2.5, 1]})

BAR_MODELS = K_ORDERED + ['Centralized RSF']
ax  = axes[0]
x_c = np.arange(N_CLIENTS)
n_m = len(BAR_MODELS)
w   = 0.16
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

for j, name in enumerate(BAR_MODELS):
    per_client_vals = [_model_folds(name, i) for i in range(N_CLIENTS)]
    positions = x_c + offsets[j]
    bp = ax.boxplot(per_client_vals, positions=positions, widths=w * 0.85,
                    patch_artist=True, showmeans=True, meanline=True, whis=(0, 100),
                    manage_ticks=False)
    _style_box(bp, COLORS[name], HATCHES[name])
    # Jittered fold points
    rng_p = np.random.default_rng(j * 101 + 7)
    for pos, vals in zip(positions, per_client_vals):
        jit = rng_p.uniform(-w * 0.18, w * 0.18, len(vals))
        ax.scatter(pos + jit, vals, s=10, color='#111111', alpha=0.5, zorder=5)
        m = float(np.nanmean(vals))
        s = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
        y_txt = float(np.nanmax(vals)) + 0.010
        _annotate_mean_std(ax, pos, y_txt, m, s, '{:.3f}', '+/-{:.3f}',
                           mean_fs=6.2, std_fs=5.0)

# Legend proxies
legend_handles = [Patch(facecolor=COLORS[n], edgecolor='#111111', hatch=HATCHES[n], label=n)
                  for n in BAR_MODELS]
ax.legend(handles=legend_handles, fontsize=7.5, ncol=2, loc='upper right')

ax.axhline(0.5, color='#888888', ls=':', linewidth=1.0)
ax.set_xticks(x_c)
ax.set_xticklabels(
    [f'C{i+1}\n(n≈{int(np.mean([len(fold_y_tests[f][i]) for f in range(N_FOLDS)]))})'
     for i in range(N_CLIENTS)])
ax.set_xlim(-0.5, N_CLIENTS - 0.5)
ax.set_ylabel('C-index (own test set)')
ax.set_title(f'(a) Per-Test-Set C-index by Model\n'
             f'(box = distribution over {N_FOLDS} CV folds  ·  labels = mean and +/-SD)')
ax.set_ylim(0.30, 1.05)

# ── Right: pooled over clients (N_FOLDS × N_CLIENTS values per model) ────────
ax2 = axes[1]
x_s = np.arange(len(BAR_MODELS))
for j, name in enumerate(BAR_MODELS):
    pooled_vals = np.concatenate([_model_folds(name, i) for i in range(N_CLIENTS)])
    bp = ax2.boxplot([pooled_vals], positions=[j], widths=0.55,
                     patch_artist=True, showmeans=True, meanline=True, whis=(0, 100),
                     manage_ticks=False)
    _style_box(bp, COLORS[name], HATCHES[name])
    rng_p = np.random.default_rng(j * 99 + 3)
    jit = rng_p.uniform(-0.12, 0.12, len(pooled_vals))
    ax2.scatter(np.full(len(pooled_vals), j) + jit, pooled_vals,
                s=10, color='#111111', alpha=0.5, zorder=5)
    m = float(np.nanmean(pooled_vals))
    s = float(np.nanstd(pooled_vals, ddof=1)) if len(pooled_vals) > 1 else np.nan
    y_txt = float(np.nanmax(pooled_vals)) + 0.012
    _annotate_mean_std(ax2, j, y_txt, m, s, '{:.3f}', '+/-{:.3f}',
                       mean_fs=9.5, std_fs=7.2)

ax2.axhline(0.5, color='#888888', ls=':', linewidth=1.0)
ax2.set_xticks(x_s)
ax2.set_xticklabels([n.replace(' (', '\n(') for n in BAR_MODELS], fontsize=8)
ax2.set_xlim(-0.6, len(BAR_MODELS) - 0.4)
ax2.set_ylabel('C-index')
ax2.set_title(f'(b) Pooled across clients\n'
              f'({N_FOLDS} folds × {N_CLIENTS} clients values per box)')
ax2.set_ylim(0.30, 1.05)

fig.suptitle(f'C-index Comparison — {N_FOLDS}-fold CV on Per-Client Test Sets',
             fontweight='bold')
plt.tight_layout()
plt.savefig('cindex_bar.pdf', bbox_inches='tight')
plt.show()


# ## 10. Survival Metrics (RMST and Risk at Time)
# 
# For each test set (client) and each federation level k,
# compute RMST and Risk@T for high/low risk groups (median split by predicted risk).

# In[46]:


# Compute metrics for every (fold, k, i) combination + centralized.
# fold_metrics[f][(k, i)] and fold_cent_metrics[f][i].
fold_metrics      = [dict() for _ in range(N_FOLDS)]
fold_cent_metrics = [dict() for _ in range(N_FOLDS)]
for f in range(N_FOLDS):
    for k in range(1, N_CLIENTS + 1):
        for i in range(N_CLIENTS):
            fold_metrics[f][(k, i)] = risk_group_metrics(
                fold_eval[f][k][i]['risk'], fold_y_tests[f][i])
    for i in range(N_CLIENTS):
        fold_cent_metrics[f][i] = risk_group_metrics(
            fold_cent_risk[f][i], fold_y_tests[f][i])


def _metric_folds(key, model_name, i):
    """Length-N_FOLDS array of a metric value for (model, client i)."""
    if model_name == 'Centralized RSF':
        return np.array([fold_cent_metrics[f][i][key] for f in range(N_FOLDS)])
    k = list(K_LABELS.keys())[list(K_LABELS.values()).index(model_name)]
    return np.array([fold_metrics[f][(k, i)][key] for f in range(N_FOLDS)])


def _short_model_label(name):
    """Compact axis labels for publication figures."""
    if name == 'Centralized RSF':
        return 'Cent. RSF'
    if name.startswith('Fed (') and name.endswith(' clients)'):
        n_clients = name[len('Fed ('):-len(' clients)')]
        return f'Fed({n_clients})'
    return name


# Print summary (fold-averaged)
print(f'\nMetrics averaged across {N_FOLDS} folds (mean ± SD):\n')
print(f'{"":5} {"":20} {"RMST Δ (d)":>16}  {"Risk@T_long Δ":>18}')
print('─' * 65)
for name in MODEL_NAMES_ALL:
    for i in range(N_CLIENTS):
        r_d = _metric_folds('rmst_diff',       name, i)
        p_d = _metric_folds('risk_long_diff',  name, i)
        print(f'  {name[:12]:<12}  C{i+1} test:  '
              f'{np.nanmean(r_d):>+8.1f} ± {np.nanstd(r_d, ddof=1):>5.1f}   '
              f'{np.nanmean(p_d):>+6.3f} ± {np.nanstd(p_d, ddof=1):>5.3f}')


# ### 10a. RMST Δ — Discrimination per Test Set × Model
#
# One box per model per client test set.
# Δ RMST = RMST(low-risk) − RMST(high-risk) in days, computed per fold.
# Box = fold distribution, center/whiskers capture spread; points are per-fold values.
# Labels above boxes show mean with smaller-font +/- SD.
# Larger positive Δ means the model separates low- from high-risk groups better.

# In[47]:


model_labels = MODEL_NAMES_ALL           # Local, Fed(2), Fed(3), Fed(4), Centralized
xlbls = [_short_model_label(n) for n in model_labels]
x = np.arange(len(model_labels))

panel_specs = []
for i in range(N_CLIENTS):
    n_te_avg = int(np.mean([len(fold_y_tests[f][i]) for f in range(N_FOLDS)]))
    panel_specs.append((i, f'C{i+1} test\n(n≈{n_te_avg})'))

panel_stats = []
for panel_id, panel_title in panel_specs:
    vals_by_model = [_metric_folds('rmst_diff', name, panel_id) for name in model_labels]
    panel_stats.append((panel_id, panel_title, vals_by_model))

all_vals = [
    vals[np.isfinite(vals)]
    for _, _, vals_by_model in panel_stats
    for vals in vals_by_model
    if np.isfinite(vals).any()
]
all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
y_span = max(1.0, float(np.max(all_vals) - np.min(all_vals)))
y_pad = max(25.0, 0.12 * y_span)
text_offset = max(6.0, 0.025 * y_span)
y_min = min(0.0, float(np.min(all_vals)) - y_pad)
y_max = float(np.max(all_vals)) + y_pad + 2.0 * text_offset

fig, axes = plt.subplots(
    1, len(panel_specs),
    figsize=(4.8 * len(panel_specs), 6.4),
    sharey=True,
    constrained_layout=True,
)

if len(panel_specs) == 1:
    axes = [axes]
for p, (panel_id, panel_title, vals_by_model) in enumerate(panel_stats):
    ax = axes[p]
    for j, name in enumerate(model_labels):
        vals = vals_by_model[j]
        vals_f = vals[np.isfinite(vals)]
        bp_vals = vals_f if vals_f.size else np.array([0.0])

        bp = ax.boxplot([bp_vals], positions=[j], widths=0.62,
                        patch_artist=True, showmeans=True, meanline=True, whis=(0, 100),
                        manage_ticks=False)
        _style_box(bp, COLORS[name], HATCHES[name])

        if vals_f.size:
            rng_p = np.random.default_rng(1000 * p + 101 * j + 7)
            jit = rng_p.uniform(-0.10, 0.10, len(vals_f))
            ax.scatter(np.full(len(vals_f), j) + jit, vals_f,
                       s=10, color='#111111', alpha=0.5, zorder=5)

            m = float(np.nanmean(vals_f))
            s = float(np.nanstd(vals_f, ddof=1)) if len(vals_f) > 1 else np.nan
            y_txt = float(np.max(vals_f)) + text_offset
            _annotate_mean_std(ax, j, y_txt, m, s, '{:+.0f}', '+/-{:.0f}',
                               mean_fs=7.0, std_fs=5.8)

    ax.axhline(0, color='#555555', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=28, ha='right', fontsize=9.0)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.set_xlim(-0.70, len(model_labels) - 0.30)
    ax.set_ylim(y_min, y_max)
    ax.set_title(panel_title, fontsize=11.0, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10.0, fontweight='bold')
    if p == 0:
        ax.set_ylabel(
            f'Δ RMST = RMST(low) − RMST(high)  [days, horizon {RMST_HORIZON:.0f} d]',
            fontsize=10.5,
            fontweight='bold',
        )

fig.suptitle(f'RMST Discrimination per Test Set × Model  ·  '
             f'Box = distribution over {N_FOLDS} CV folds  ·  '
             f'Red line = box mean  ·  dots = per-fold values\n'
             f'Labels above boxes = mean with smaller +/-SD text',
             fontsize=13.0, fontweight='bold')
plt.savefig('rmst.pdf', bbox_inches='tight')
plt.show()


# ### 10b. Risk-at-Time Δ — Discrimination at T₁ / T₂ / T₃
#
# Grid: rows = client test sets, cols = T₁ / T₂ / T₃.
# One box per model showing Δ Risk@T = Risk(high) − Risk(low)
# (cumulative-incidence difference between model-defined risk groups).
# Box = fold distribution; dots are per-fold values.
# Labels above boxes show mean with smaller-font +/- SD.

# In[48]:


horizons = [
    ('T₁ (short)', 'risk_short', T_SHORT),
    ('T₂ (med)',   'risk_med',   T_MED),
    ('T₃ (long)',  'risk_long',  T_LONG),
]

model_labels_rt = MODEL_NAMES_ALL
xlbls_rt = [_short_model_label(n) for n in model_labels_rt]
x_rt = np.arange(len(model_labels_rt))

n_rows_rt = N_CLIENTS
fig, axes = plt.subplots(
    n_rows_rt, 3,
    figsize=(15.0, 3.2 * n_rows_rt),
    sharey='row',
    constrained_layout=True,
)

if n_rows_rt == 1:
    axes = np.array([axes])

row_specs = [(i, f'C{i+1} test') for i in range(N_CLIENTS)]

for row, (row_id, row_lbl) in enumerate(row_specs):
    col_stats = []
    row_vals = []
    for label, key, t_val in horizons:
        vals_by_model = [_metric_folds(f'{key}_diff', name, row_id) for name in model_labels_rt]
        col_stats.append((label, t_val, vals_by_model))
        for vals in vals_by_model:
            vals_f = vals[np.isfinite(vals)]
            if vals_f.size:
                row_vals.append(vals_f)

    row_all = np.concatenate(row_vals) if row_vals else np.array([0.0])
    row_span = max(1e-3, float(np.max(row_all) - np.min(row_all)))
    row_pad = max(0.025, 0.22 * row_span)
    y_txt_offset = max(0.004, 0.035 * row_span)
    y_min = min(-0.02, float(np.min(row_all)) - row_pad)
    y_max = max(0.02, float(np.max(row_all)) + row_pad + 2.0 * y_txt_offset)

    for col, (label, t_val, vals_by_model) in enumerate(col_stats):
        ax = axes[row, col]
        for j, name in enumerate(model_labels_rt):
            vals = vals_by_model[j]
            vals_f = vals[np.isfinite(vals)]
            bp_vals = vals_f if vals_f.size else np.array([0.0])

            bp = ax.boxplot([bp_vals], positions=[j], widths=0.62,
                            patch_artist=True, showmeans=True, meanline=True, whis=(0, 100),
                            manage_ticks=False)
            _style_box(bp, COLORS[name], HATCHES[name])

            if vals_f.size:
                rng_p = np.random.default_rng(10000 * row + 1000 * col + 101 * j + 3)
                jit = rng_p.uniform(-0.10, 0.10, len(vals_f))
                ax.scatter(np.full(len(vals_f), j) + jit, vals_f,
                           s=9, color='#111111', alpha=0.5, zorder=5)

                m = float(np.nanmean(vals_f))
                s = float(np.nanstd(vals_f, ddof=1)) if len(vals_f) > 1 else np.nan
                y_txt = float(np.max(vals_f)) + y_txt_offset
                _annotate_mean_std(ax, j, y_txt, m, s, '{:+.2f}', '+/-{:.2f}',
                                   mean_fs=6.8, std_fs=5.5)

        ax.axhline(0, color='#555555', lw=0.9)
        ax.set_xlim(-0.70, len(model_labels_rt) - 0.30)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=8.8)

        ax.set_xticks(x_rt)
        if row == n_rows_rt - 1:
            ax.set_xticklabels(xlbls_rt, rotation=25, ha='right', fontsize=8.8)
            ax.set_xlabel('Model', fontsize=10.0, fontweight='bold')
        else:
            ax.set_xticklabels([])

        if col == 0:
            y_label = f'C{row_id+1}\nΔ P(T ≤ t)'
            ax.set_ylabel(y_label, fontsize=9.8, fontweight='bold')
        if row == 0:
            ax.set_title(f'{label}\nt = {t_val:.0f} d ({t_val/365:.1f} yr)',
                         fontsize=11.0, fontweight='bold')

fig.suptitle(f'Risk-at-Time Discrimination — Δ P(T ≤ t) = Risk(high) − Risk(low)\n'
             f'Box = distribution over {N_FOLDS} CV folds  ·  '
             f'Red line = box mean  ·  dots = per-fold values\n'
             f'Labels above boxes = mean with smaller +/-SD text',
             fontsize=13.0, fontweight='bold')
plt.savefig('risk_at_time.pdf', bbox_inches='tight')
plt.show()


# ## 11. Kaplan–Meier Curves
# 
# One panel per client test set.
# Solid lines = Local model, dashed = Fed (4 clients).
# Blue = low-risk, Red = high-risk (median split by predicted risk).

# In[49]:


# Show 3 models: Local (solid), Fed(4) (dashed), Centralized (dash-dot-dot)
KM_MODELS = [
    (1,              K_LABELS[1],      '-',                2.0, 0.85),
    (N_CLIENTS,      K_LABELS[N_CLIENTS], '--',            2.0, 0.85),
    ('centralized',  'Centralized RSF', (0,(3,1,1,1)),     2.0, 0.85),
]

fig, outer_axes = plt.subplots(
    2, N_CLIENTS,
    figsize=(5 * N_CLIENTS, 9),
    gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08},
)
nar_times = np.array([0, T_SHORT, T_MED, T_LONG])
t_grid    = np.linspace(0.0, float(RMST_HORIZON), 200)


def _fold_risk(k_id, f, i):
    """Risk scores for a given (model, fold, client)."""
    return fold_cent_risk[f][i] if k_id == 'centralized' else fold_eval[f][k_id][i]['risk']


def _km_on_grid(events, times, grid):
    """Step-interpolate KM survival onto a common time grid; returns s(grid)."""
    if events.sum() < 1:
        return None
    km_t, km_s = kaplan_meier_estimator(events, times)
    if len(km_t) == 0:
        return None
    # Step ('previous') interpolation: pre-first-event = 1.0, past-last kept constant.
    f_ = interp1d(km_t, km_s, kind='previous', bounds_error=False,
                  fill_value=(1.0, float(km_s[-1])))
    return f_(grid)


for col, i in enumerate(range(N_CLIENTS)):
    ax_km  = outer_axes[0, col]
    ax_nar = outer_axes[1, col]

    for k_id, model_name, ls, lw, alpha in KM_MODELS:
        for grp_name, color in [('Low', C_LOW), ('High', C_HIGH)]:
            curves_on_grid = []
            for f in range(N_FOLDS):
                y_te   = fold_y_tests[f][i]
                events = y_te[event_field]
                times  = y_te[time_field]
                risk   = _fold_risk(k_id, f, i)
                mask   = (risk >= np.median(risk)) if grp_name == 'High' \
                         else ~(risk >= np.median(risk))
                e_, t_ = events[mask], times[mask]
                if e_.sum() < 1:
                    continue
                km_t, km_s = kaplan_meier_estimator(e_, t_)
                # Faded per-fold step curve
                ax_km.step(km_t, km_s, where='post', color=color,
                           lw=lw * 0.6, ls=ls, alpha=0.22, zorder=3)
                s_grid = _km_on_grid(e_, t_, t_grid)
                if s_grid is not None:
                    curves_on_grid.append(s_grid)
            if curves_on_grid:
                mean_curve = np.mean(np.vstack(curves_on_grid), axis=0)
                lbl = (f'{model_name} — {grp_name} risk' if col == 0 else None)
                ax_km.plot(t_grid, mean_curve, color=color, lw=lw, ls=ls,
                           alpha=alpha, label=lbl, zorder=5)

    # Annotation: C-index mean ± SD across folds for all 3 models
    c_loc  = np.array([fold_eval[f][1][i]['c_index']          for f in range(N_FOLDS)])
    c_fed  = np.array([fold_eval[f][N_CLIENTS][i]['c_index']  for f in range(N_FOLDS)])
    c_cent = np.array([fold_cent_c[f][i]                      for f in range(N_FOLDS)])
    ann_lines = [
        f'{K_LABELS[1]}: C={c_loc.mean():.3f}±{c_loc.std(ddof=1):.3f}',
        f'{K_LABELS[N_CLIENTS]}: C={c_fed.mean():.3f}±{c_fed.std(ddof=1):.3f}',
        f'Centralized: C={c_cent.mean():.3f}±{c_cent.std(ddof=1):.3f}',
    ]
    ax_km.text(0.97, 0.97, '\n'.join(ann_lines), transform=ax_km.transAxes,
               ha='right', va='top', fontsize=7.5,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#bbbbbb', alpha=0.9))

    # Log-rank p-value: mean across folds using Fed(N) risk split
    pvals = []
    for f in range(N_FOLDS):
        try:
            y_te_f = fold_y_tests[f][i]
            risk_ref = fold_eval[f][N_CLIENTS][i]['risk']
            _, pv = compare_survival(y_te_f, (risk_ref >= np.median(risk_ref)).astype(int))
            pvals.append(pv)
        except Exception:
            pass
    if pvals:
        pv_mean = float(np.mean(pvals))
        stars = ('***' if pv_mean < 0.001 else
                 '**'  if pv_mean < 0.01  else
                 '*'   if pv_mean < 0.05  else 'ns')
        ax_km.text(0.97, 0.50,
                   f'Log-rank p̄ = {pv_mean:.4f} {stars}',
                   transform=ax_km.transAxes, ha='right', va='top', fontsize=8.5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#bbbbbb', alpha=0.9))

    for t_mark, lbl in [(T_SHORT, 'T₁'), (T_MED, 'T₂'), (T_LONG, 'T₃')]:
        ax_km.axvline(t_mark, color='#888888', ls=':', lw=1.0)
        ax_km.text(t_mark, 1.03, lbl, ha='center', fontsize=8.5,
                   transform=ax_km.get_xaxis_transform(), color='#555555')
    ax_km.axvline(RMST_HORIZON, color='#555555', ls='--', lw=1.2, label='RMST horizon')

    n_te_avg = int(np.mean([len(fold_y_tests[f][i]) for f in range(N_FOLDS)]))
    ev_avg   = np.mean([fold_y_tests[f][i][event_field].sum() for f in range(N_FOLDS)])
    ax_km.set_title(f'C{i+1}  (n≈{n_te_avg}, ev≈{ev_avg:.1f})', fontweight='bold')
    ax_km.set_ylim(-0.02, 1.12)
    ax_km.set_xlim(0, float(RMST_HORIZON))
    ax_km.set_ylabel('Survival probability' if col == 0 else '')
    ax_km.tick_params(labelbottom=False)

    # n-at-risk panel — averaged across folds, Fed(N) risk split
    ax_nar.set_facecolor('white')
    ax_nar.set_xlim(ax_km.get_xlim())
    ax_nar.set_ylim(-0.5, 1.5)
    ax_nar.set_yticks([0, 1])
    ax_nar.set_yticklabels(['High', 'Low'], fontsize=8.5)
    ax_nar.tick_params(left=False, bottom=False)
    ax_nar.grid(False)
    ax_nar.spines[['top', 'right', 'left']].set_visible(False)
    for row_pos, grp_name, color in [(0, 'High', C_HIGH), (1, 'Low', C_LOW)]:
        for t_q in nar_times:
            counts = []
            for f in range(N_FOLDS):
                y_te_f   = fold_y_tests[f][i]
                times_f  = y_te_f[time_field]
                risk_f   = fold_eval[f][N_CLIENTS][i]['risk']
                high_f   = risk_f >= np.median(risk_f)
                mask_f   = high_f if grp_name == 'High' else ~high_f
                counts.append(int(np.sum(times_f[mask_f] >= t_q)))
            ax_nar.text(t_q, row_pos, f'{np.mean(counts):.0f}',
                        ha='center', va='center', fontsize=8.5, color=color)
    ax_nar.set_xlabel('Time (days)')
    if col == 0:
        ax_nar.text(-0.22, 0.5, 'n at risk\n(Fed, avg folds)',
                    transform=ax_nar.transAxes,
                    fontsize=8, va='center', ha='right', style='italic')

# Single legend
legend_handles = (
    [Line2D([0],[0], color='#555555', ls=ls, lw=1.8, label=name)
     for _, name, ls, _, _ in KM_MODELS] +
    [Line2D([0],[0], color=C_HIGH, lw=2.0, label='High-risk'),
     Line2D([0],[0], color=C_LOW,  lw=2.0, label='Low-risk')]
)
outer_axes[0, 0].legend(handles=legend_handles, fontsize=8.5, loc='lower left')

fig.suptitle(f'Kaplan–Meier Curves — Local vs Fed ({N_CLIENTS} clients) vs Centralized\n'
             f'Bold = mean over {N_FOLDS} CV folds  ·  faded = per-fold steps  ·  '
             f'Solid = Local  ·  Dashed = Fed ({N_CLIENTS})  ·  Dash-dot = Centralized  ·  '
             f'Color = risk group',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('km_curves.pdf', bbox_inches='tight')
plt.show()


# ## 12. Feature Importance — Final Federation (k = N_CLIENTS), fold 0
#
# Permutation importance for each client's federated model on its own test set.
# Uses fold 0 only (single representative model) to avoid multiplying compute by N_FOLDS.

# In[50]:

# DISABELED
# TOP_N = 12

# client_imps   = {}
# all_feat_pool = set()

# print(f'Computing permutation importance for k={N_CLIENTS} federation (fold 0) …')
# models_final = [copy.deepcopy(local_models[i]) for i in range(N_CLIENTS)]
# fed_final = FederatedRandomSurvivalForest(local_models=models_final)
# fed_final.distribute_trees()

# for i in range(N_CLIENTS):
#     n_cross = len(models_final[i]._federated_estimators)
#     if n_cross == 0:
#         print(f'  C{i+1}: no cross-client trees, skipping importance')
#         continue
#     models_final[i].use_federated_estimators()
#     X_te = X_client_tests[i]
#     perm = permutation_importance(
#         models_final[i], X_te, y_client_tests[i],
#         n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
#     )
#     models_final[i].use_local_estimators()
#     client_imps[i] = dict(zip(X_te.columns, perm.importances_mean))
#     all_feat_pool.update(X_te.columns)
#     print(f'  C{i+1}: {n_cross} cross-client trees, {len(X_te.columns)} features')

# if client_imps:
#     feat_max  = {f: max(client_imps[i].get(f, 0.0) for i in client_imps) for f in all_feat_pool}
#     top_feats = sorted(feat_max, key=feat_max.get, reverse=True)[:TOP_N]

#     n_cli  = len(client_imps)
#     y_pos  = np.arange(TOP_N)
#     w_bar  = 0.72 / n_cli

#     fig, ax = plt.subplots(figsize=(10, 6))
#     for j, i in enumerate(sorted(client_imps)):
#         vals   = [client_imps[i].get(f, 0.0) for f in top_feats]
#         offset = (j - n_cli / 2.0 + 0.5) * w_bar
#         ax.barh(y_pos + offset, vals, height=w_bar * 0.88,
#                 color=CLIENT_COLORS[j], label=f'C{i+1}',
#                 edgecolor='#333333', linewidth=0.5)

#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(top_feats, fontsize=9)
#     ax.invert_yaxis()
#     ax.axvline(0, color='#555555', lw=0.8)
#     ax.set_xlabel('Permutation importance (decrease in C-index)', fontsize=10)
#     ax.set_title(f'Top {TOP_N} Features — Fed ({N_CLIENTS} clients)\n'
#                  f'Each client evaluated on its own test set with cross-client trees',
#                  fontweight='bold')
#     ax.legend(fontsize=9, loc='lower right')
#     plt.tight_layout()
#     plt.savefig('feature_importance.pdf', bbox_inches='tight')
#     plt.show()
# else:
#     print('No federated importance to plot.')

