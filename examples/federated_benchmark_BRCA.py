#!/usr/bin/env python
# coding: utf-8

# # Federated RSF — Per-Client Convergence Benchmark with Repeated Random Site Splits (BRCA)
# 
# For each of `N_SITE_SPLITS` random patient-to-client partitions:
# 1. Split patients across `N_CLIENTS` non-overlapping cohorts and drop a fraction of features per client.
# 2. Run `N_FOLDS`-fold stratified cross-validation per client.
# 3. For each fold, fit local models, Centralized-SRF (feature-dropped) RSF, and a complete-feature centralized oracle.
# 4. Evaluate cumulative federation `Fed(k)` for `k=1..N_CLIENTS` and the all-subset enumeration used for the convergence plot.
# 
# All summary tables and plots aggregate across `N_SITE_SPLITS × N_FOLDS` evaluations per (model, client).
# 
# | Model | Description |
# |-------|-------------|
# | **Local** | Trained only on that client's data |
# | **Fed (k)** | Federated cumulatively across the first `k` clients |
# | **Fed (N)** | Federated across all `N_CLIENTS` clients |
# | **Centralized-SRF** | Pooled feature-dropped data, NaN for absent features |
# | **Centralized** | Pooled complete-feature data — discrimination ceiling |
# 
# **Metrics:** C-index, RMST Δ, Risk at Time, Kaplan–Meier curves.

# In[67]:


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
from scipy import stats
from scipy.interpolate import interp1d

from sksurv.datasets import load_gbsg2 as load_data
from sksurv.ensemble import RandomSurvivalForest
from sksurv.exceptions import NoComparablePairException
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival

try:
    from federated_rsf.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
    from federated_rsf.schema import DatasetSchema, SchemaAligner, SchemaCreator
except ModuleNotFoundError as exc:
    if exc.name != 'federated_rsf':
        raise
    project_root = Path.cwd().resolve()
    while project_root != project_root.parent:
        if (project_root / 'src' / 'federated_rsf').is_dir():
            break
        project_root = project_root.parent
    for p in (project_root / 'src', project_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    from federated_rsf.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
    from federated_rsf.schema import DatasetSchema, SchemaAligner, SchemaCreator

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
# 
# `N_SITE_SPLITS` controls the number of random patient-to-client partitions.
# Each partition gets its own per-client `N_FOLDS` cross-validation, and all
# metrics aggregate across the `N_SITE_SPLITS × N_FOLDS` evaluations.

# In[68]:


N_CLIENTS           = 10      # hospital sites
N_SITE_SPLITS       = 5       # repeated random patient-to-client partitions
DROP_FEATURE_PCT    = 0.35    # fraction of features dropped per client
N_FOLDS             = 5       # stratified CV folds per client per site split
N_ESTIMATORS        = 300     # trees per local model
RANDOM_STATE        = 999
UPDATE_METHOD       = 'constant'
MIN_FEATURE_OVERLAP = 1.0     # 1.0 = strict subset (recipient must have every tree feature)

K_LABELS = {1: 'Local'} | {
    k: f'Fed ({k} clients)' for k in range(2, N_CLIENTS + 1)
}
K_ORDERED = [K_LABELS[k] for k in range(1, N_CLIENTS + 1)]
MODEL_NAMES_ALL = K_ORDERED + ['Centralized-SRF', 'Centralized']

_fed_shades = np.linspace(35, 185, max(N_CLIENTS, 2)).astype(int)
COLORS = {K_LABELS[1]: '#111111'}
for k in range(2, N_CLIENTS + 1):
    shade = _fed_shades[k - 1]
    COLORS[K_LABELS[k]] = f'#{shade:02x}{shade:02x}{shade:02x}'
COLORS['Centralized-SRF'] = '#0A7F78'
COLORS['Centralized'] = '#005599'

_hatch_cycle = ['///', '...', 'xxx', '\\\\\\', '---', '+++', 'ooo', '***', '|||']
HATCHES = {K_LABELS[1]: ''}
for k in range(2, N_CLIENTS + 1):
    HATCHES[K_LABELS[k]] = _hatch_cycle[(k - 2) % len(_hatch_cycle)]
HATCHES['Centralized-SRF'] = '\\\\\\'
HATCHES['Centralized'] = '---'

_client_marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
CLIENT_COLORS = [plt.get_cmap('tab10')(i % 10) for i in range(N_CLIENTS)]
CLIENT_MARKERS = [_client_marker_cycle[i % len(_client_marker_cycle)] for i in range(N_CLIENTS)]
C_HIGH = '#CC2222'
C_LOW  = '#2266CC'

print(f'Clients: {N_CLIENTS} | Site splits: {N_SITE_SPLITS} | Drop: {DROP_FEATURE_PCT*100:.0f}% | '
      f'CV folds: {N_FOLDS} | Trees: {N_ESTIMATORS}')


# ## 2. Load Dataset

# In[69]:


X_raw, y_raw = load_data()
# Diagnostic snapshot only — per-client OHE is applied after partitioning so
# every client retains a consistent set of indicator columns for any kept
# categorical (avoids silent zero-imputation across sites).
X_full = OneHotEncoder().fit_transform(X_raw)

event_field = y_raw.dtype.names[0]
time_field  = y_raw.dtype.names[1]

all_times   = y_raw[time_field]
event_times = all_times[y_raw[event_field]]

T_SHORT      = float(np.percentile(event_times, 25))
T_MED        = float(np.percentile(event_times, 50))
T_LONG       = float(np.percentile(event_times, 75))
RMST_HORIZON = float(np.percentile(event_times, 80))

CENT_N_EST = N_CLIENTS * N_ESTIMATORS

print(f'Samples : {len(X_full)}   Features: {X_full.shape[1]}')
print(f'Events  : {y_raw[event_field].sum()} ({y_raw[event_field].mean()*100:.1f}%)')
print(f'Follow-up: {all_times.min():.0f}–{all_times.max():.0f} d  '
      f'(median {np.median(all_times):.0f} d = {np.median(all_times)/365:.1f} yr)')
print(f'Time horizons (days):  T_short={T_SHORT:.0f}  T_med={T_MED:.0f}  '
      f'T_long={T_LONG:.0f}  RMST horizon={RMST_HORIZON:.0f}')


# ## 3. Helper Functions

# In[81]:


def c_idx(risk, y):
    try:
        return concordance_index_censored(y[event_field], y[time_field], risk)[0]
    except (NoComparablePairException, ValueError):
        return np.nan


def compute_rmst(events, times, horizon):
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
    try:
        km_t, km_s = kaplan_meier_estimator(events, times)
        idx  = np.searchsorted(km_t, t_query, side='right') - 1
        surv = km_s[max(0, idx)] if idx >= 0 else 1.0
        return float(1.0 - surv)
    except Exception:
        return np.nan


def risk_group_metrics(risk_scores, y):
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


def _fed_combined_predict(model, X):
    if len(model._federated_estimators) == 0:
        model.use_local_estimators()
        return model.predict(X)
    model.use_federated_estimators()
    risk = model.predict(X)
    model.use_local_estimators()
    return risk


def _assert_federated_tree_consistency(fed_model):
    all_features = set(fed_model.all_features)
    estimator_pos = {id(estimator): j for j, estimator in enumerate(fed_model.estimators_)}
    for tree_idx, feat_set in enumerate(fed_model.tree_features):
        if not feat_set.issubset(all_features):
            missing = sorted(feat_set - all_features)
            raise AssertionError(
                f'Tree {tree_idx} references features absent from the global schema: {missing}'
            )
    for model_idx, model in enumerate(fed_model.local_models):
        for estimator in model._federated_estimators:
            estimator_idx = estimator_pos.get(id(estimator))
            if estimator_idx is None:
                raise AssertionError(f'C{model_idx+1} received a tree outside the federated pool')
            feat_set = fed_model.tree_features[estimator_idx]
            if MIN_FEATURE_OVERLAP >= 1.0 and not feat_set.issubset(model.local_features):
                raise AssertionError(
                    f'C{model_idx+1} received an incompatible tree: '
                    f'{sorted(feat_set - model.local_features)} missing features'
                )


def _short_model_label(name):
    if name == 'Centralized-SRF':
        return 'Centralized-SRF'
    if name == 'Centralized':
        return 'Centralized'
    if name.startswith('Fed (') and name.endswith(' clients)'):
        n_clients = name[len('Fed ('):-len(' clients)')]
        return f'Fed({n_clients})'
    return name

def _annotate_mean_std(ax, x, y, mean, std, mean_fmt, std_fmt,
                       mean_fs=9.5, std_fs=8.0):
    """Annotate (mean, std) above a data point: bold mean at (x, y), lighter ±std stacked above."""
    from matplotlib.transforms import offset_copy
    ax.text(x, y, mean_fmt.format(mean),
            ha='center', va='bottom',
            fontsize=mean_fs, fontweight='bold',
            clip_on=True)
    tr = offset_copy(ax.transData, fig=ax.figure, x=0, y=mean_fs + 1, units='points')
    ax.text(x, y, std_fmt.format(std),
            ha='center', va='bottom',
            fontsize=std_fs, color='#555555',
            transform=tr, clip_on=True)



# ## 4. Per-Site-Split Data Construction
# 
# For each site partition seed:
# - Shuffle patients and split into `N_CLIENTS` non-overlapping cohorts.
# - Drop a `DROP_FEATURE_PCT` fraction of (pre-OHE) features per client.
# - One-hot encode each client *after* splitting and dropping.
# - Align all client tables to the global feature union (NaN for absent features).
# - Build per-client stratified folds on the event indicator (fall back to `KFold` for tiny classes).

# In[71]:


def _client_sizes(n_samples, n_clients):
    sizes = [n_samples // n_clients] * (n_clients - 1)
    sizes.append(n_samples - sum(sizes))
    return sizes


def make_site_split(site_seed):
    """Build per-client feature-dropped + complete-feature views for one site split."""
    rng = np.random.default_rng(site_seed)
    positions = np.arange(len(X_raw))
    rng.shuffle(positions)

    X_clients_pre_ohe = []
    X_clients_full   = []
    y_clients        = []
    start = 0
    for client_size in _client_sizes(len(X_raw), N_CLIENTS):
        pos = positions[start:start + client_size]
        start += client_size
        X_site = X_raw.iloc[pos].reset_index(drop=True)
        n_drop = int(round(len(X_site.columns) * DROP_FEATURE_PCT))
        drop_cols = rng.choice(X_site.columns.to_numpy(), size=n_drop, replace=False)
        X_clients_pre_ohe.append(X_site.drop(columns=drop_cols))
        X_clients_full.append(X_full.iloc[pos].reset_index(drop=True))
        y_clients.append(y_raw[pos])

    X_clients_raw = [OneHotEncoder().fit_transform(X_c) for X_c in X_clients_pre_ohe]
    client_feat_sets = [sorted(X_c.columns.tolist()) for X_c in X_clients_raw]

    schema_list = [DatasetSchema(cols) for cols in client_feat_sets]
    federated_schemas = SchemaCreator(anonymize=False).fit_transform(schema_list)
    global_columns = federated_schemas[0].columns

    return {
        'X_clients_raw'   : X_clients_raw,
        'X_clients_full'  : X_clients_full,
        'y_clients'       : y_clients,
        'client_feat_sets': client_feat_sets,
        'global_columns'  : global_columns,
    }


def make_fold_splits(X_clients_raw, y_clients, split_seed):
    fold_splits = []
    for i, y_c in enumerate(y_clients):
        events_c = y_c[event_field].astype(int)
        n_pos = int(events_c.sum())
        n_neg = int(len(events_c) - n_pos)
        if min(n_pos, n_neg) >= N_FOLDS:
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
            splits_i = list(splitter.split(X_clients_raw[i], events_c))
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
            splits_i = list(splitter.split(X_clients_raw[i]))
        fold_splits.append(splits_i)
    return fold_splits


# ## 5. Main Loop — Site Splits × CV Folds
# 
# Outer loop: `N_SITE_SPLITS` random patient partitions.
# Inner loop: `N_FOLDS` per-client cross-validation folds.
# 
# Per (site_split, fold) we fit:
# - One `LocalRandomSurvivalForest` per client.
# - Cumulative federation `Fed(k)` for `k=1..N_CLIENTS`.
# - All-subset enumeration: every non-empty subset `S ⊆ {C1..CN}` is federated, every member evaluated on its own test set.
# - Centralized-SRF on the feature-dropped union (NaN for absent features).
# - Centralized on the complete (pre-drop) features — discrimination ceiling.
# 
# All containers are double-indexed `[site_split][fold]`.

# In[72]:


site_seeds = [RANDOM_STATE + 10_000 * s for s in range(N_SITE_SPLITS)]

fold_X_trains      = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_X_tests       = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_y_trains      = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_y_tests       = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_local_models  = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_fair_c        = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_fair_risk     = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_cent_c        = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_cent_risk     = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_eval          = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
fold_subset_pc     = [[None]*N_FOLDS for _ in range(N_SITE_SPLITS)]
site_global_columns = [None]*N_SITE_SPLITS
site_client_feat_sets = [None]*N_SITE_SPLITS

for s, site_seed in enumerate(site_seeds):
    print(f'\n████████ Site split {s+1}/{N_SITE_SPLITS} (seed={site_seed}) ████████')
    bundle = make_site_split(site_seed)
    X_clients_raw    = bundle['X_clients_raw']
    X_clients_full   = bundle['X_clients_full']
    y_clients        = bundle['y_clients']
    client_feat_sets = bundle['client_feat_sets']
    global_columns   = bundle['global_columns']
    site_global_columns[s]   = global_columns
    site_client_feat_sets[s] = client_feat_sets

    fold_splits = make_fold_splits(X_clients_raw, y_clients, site_seed)

    for f in range(N_FOLDS):
        print(f'  ── Fold {f+1}/{N_FOLDS} ──')
        X_trs, X_tes, y_trs, y_tes = [], [], [], []
        X_fair_tes = []
        X_full_trs, X_full_tes = [], []
        for i in range(N_CLIENTS):
            tr_idx, te_idx = fold_splits[i][f]
            X_c = X_clients_raw[i][client_feat_sets[i]]
            X_tr = X_c.iloc[tr_idx].reset_index(drop=True).reindex(columns=global_columns)
            X_te = X_c.iloc[te_idx].reset_index(drop=True).reindex(columns=global_columns)
            X_trs.append(X_tr)
            X_tes.append(X_te)
            X_fair_tes.append(X_te)
            y_trs.append(y_clients[i][tr_idx])
            y_tes.append(y_clients[i][te_idx])
            X_full_trs.append(X_clients_full[i].iloc[tr_idx].reset_index(drop=True))
            X_full_tes.append(X_clients_full[i].iloc[te_idx].reset_index(drop=True))
        fold_X_trains[s][f] = X_trs
        fold_X_tests[s][f]  = X_tes
        fold_y_trains[s][f] = y_trs
        fold_y_tests[s][f]  = y_tes

        local_models_f = []
        for i in range(N_CLIENTS):
            m = LocalRandomSurvivalForest(
                n_estimators=N_ESTIMATORS, random_state=site_seed + f,
                update_method=UPDATE_METHOD, min_samples_split=6, min_samples_leaf=3,
            )
            m.fit(X_trs[i], y_trs[i])
            local_models_f.append(m)
        fold_local_models[s][f] = local_models_f

        # Centralized-SRF (feature-dropped, NaN for absent)
        X_fair_train = pd.concat(X_trs, ignore_index=True)
        y_fair_train = np.concatenate(y_trs)
        fair_pooled_model = RandomSurvivalForest(
            n_estimators=CENT_N_EST, random_state=site_seed + f,
            min_samples_split=6, min_samples_leaf=3,
        )
        fair_pooled_model.fit(X_fair_train, y_fair_train)
        fair_risk_f, fair_c_f = {}, {}
        for i in range(N_CLIENTS):
            r = fair_pooled_model.predict(X_fair_tes[i])
            fair_risk_f[i], fair_c_f[i] = r, c_idx(r, y_tes[i])
        fold_fair_c[s][f]    = fair_c_f
        fold_fair_risk[s][f] = fair_risk_f

        # Centralized oracle (complete features)
        X_cent_train = pd.concat(X_full_trs, ignore_index=True)
        y_cent_train = np.concatenate(y_trs)
        centralized_model = RandomSurvivalForest(
            n_estimators=CENT_N_EST, random_state=site_seed + f,
            min_samples_split=6, min_samples_leaf=3,
        )
        centralized_model.fit(X_cent_train, y_cent_train)
        if len(centralized_model.estimators_) != CENT_N_EST:
            raise AssertionError('Centralized fitted an unexpected number of trees')
        if list(centralized_model.feature_names_in_) != list(X_cent_train.columns):
            raise AssertionError('Centralized feature_names_in_ differ from training columns')
        cent_risk_f, cent_c_f = {}, {}
        for i in range(N_CLIENTS):
            if list(X_full_tes[i].columns) != list(centralized_model.feature_names_in_):
                raise AssertionError(f'C{i+1} centralized test columns do not match training')
            r = centralized_model.predict(X_full_tes[i])
            cent_risk_f[i], cent_c_f[i] = r, c_idx(r, y_tes[i])
        fold_cent_c[s][f]    = cent_c_f
        fold_cent_risk[s][f] = cent_risk_f

        # Cumulative federation k=1..N_CLIENTS
        eval_f = {}
        for k in range(1, N_CLIENTS + 1):
            models_k = [copy.deepcopy(local_models_f[i]) for i in range(N_CLIENTS)]
            if k >= 2:
                fed = FederatedRandomSurvivalForest(local_models=models_k[:k])
                fed.distribute_trees(min_feature_overlap=MIN_FEATURE_OVERLAP)
                _assert_federated_tree_consistency(fed)
            eval_f[k] = {}
            for i in range(N_CLIENTS):
                n_cross = len(models_k[i]._federated_estimators)
                risk = _fed_combined_predict(models_k[i], X_tes[i])
                eval_f[k][i] = {'c_index': c_idx(risk, y_tes[i]), 'risk': risk, 'n_fed_trees': n_cross}
        fold_eval[s][f] = eval_f

        # All-subset enumeration
        subset_pc_f = {}
        for k in range(1, N_CLIENTS + 1):
            for subset in _combinations(range(N_CLIENTS), k):
                subset_models = [copy.deepcopy(local_models_f[i]) for i in subset]
                if k >= 2:
                    fed_sub = FederatedRandomSurvivalForest(local_models=subset_models)
                    fed_sub.distribute_trees(min_feature_overlap=MIN_FEATURE_OVERLAP)
                    _assert_federated_tree_consistency(fed_sub)
                fs = frozenset(subset)
                for j, i in enumerate(subset):
                    risk = _fed_combined_predict(subset_models[j], X_tes[i])
                    subset_pc_f[(fs, i)] = c_idx(risk, y_tes[i])
        fold_subset_pc[s][f] = subset_pc_f

        loc_mean  = np.nanmean([eval_f[1][i]['c_index'] for i in range(N_CLIENTS)])
        fed_mean  = np.nanmean([eval_f[N_CLIENTS][i]['c_index'] for i in range(N_CLIENTS)])
        fair_mean = np.nanmean(list(fair_c_f.values()))
        cent_mean = np.nanmean(list(cent_c_f.values()))
        print(f'    Local={loc_mean:.4f}  Fed({N_CLIENTS})={fed_mean:.4f}  '
              f'Fair={fair_mean:.4f}  Centralized={cent_mean:.4f}')

print('\nAll site splits + folds complete.')


# ## 6. Aggregation Helpers
# 
# All downstream summaries pool across `N_SITE_SPLITS × N_FOLDS = {N_SITE_SPLITS}×{N_FOLDS}` evaluations per (model, client).

# In[73]:


SF_PAIRS = [(s, f) for s in range(N_SITE_SPLITS) for f in range(N_FOLDS)]
N_SF = len(SF_PAIRS)


def _model_folds(name, i):
    """Length-N_SF array of C-index for (model, client i) across all (site, fold) pairs."""
    if name == 'Centralized-SRF':
        return np.array([fold_fair_c[s][f][i] for s, f in SF_PAIRS])
    if name == 'Centralized':
        return np.array([fold_cent_c[s][f][i] for s, f in SF_PAIRS])
    k = next(kk for kk, lbl in K_LABELS.items() if lbl == name)
    return np.array([fold_eval[s][f][k][i]['c_index'] for s, f in SF_PAIRS])


def _model_metric_folds(key, name, i):
    """Length-N_SF array of risk_group_metrics[key] for (model, client i)."""
    out = np.empty(N_SF)
    for idx, (s, f) in enumerate(SF_PAIRS):
        if name == 'Centralized-SRF':
            r = fold_fair_risk[s][f][i]
        elif name == 'Centralized':
            r = fold_cent_risk[s][f][i]
        else:
            k = next(kk for kk, lbl in K_LABELS.items() if lbl == name)
            r = fold_eval[s][f][k][i]['risk']
        out[idx] = risk_group_metrics(r, fold_y_tests[s][f][i])[key]
    return out


def _pooled_cindex(name, s, f):
    """C-index pooled across clients for one (site, fold) pair."""
    if name == 'Centralized-SRF':
        risks = np.concatenate([fold_fair_risk[s][f][i] for i in range(N_CLIENTS)])
    elif name == 'Centralized':
        risks = np.concatenate([fold_cent_risk[s][f][i] for i in range(N_CLIENTS)])
    else:
        k = next(kk for kk, lbl in K_LABELS.items() if lbl == name)
        risks = np.concatenate([fold_eval[s][f][k][i]['risk'] for i in range(N_CLIENTS)])
    y_all = np.concatenate([fold_y_tests[s][f][i] for i in range(N_CLIENTS)])
    return c_idx(risks, y_all)

def _style_box(bp, color, hatch=''):
    """Apply consistent fill / hatch / edge styling to a matplotlib boxplot dict."""
    for box in bp.get('boxes', []):
        box.set_facecolor(color)
        box.set_edgecolor('black')
        box.set_linewidth(1.0)
        if hatch:
            box.set_hatch(hatch)
    for whisker in bp.get('whiskers', []):
        whisker.set_color('black')
        whisker.set_linewidth(1.0)
    for cap in bp.get('caps', []):
        cap.set_color('black')
        cap.set_linewidth(1.0)
    for median in bp.get('medians', []):
        median.set_color('black')
        median.set_linewidth(1.5)
    for flier in bp.get('fliers', []):
        flier.set_marker('o')
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor('black')
        flier.set_markersize(3)
        flier.set_alpha(0.6)


# ## 7. C-index Summary Table
# 
# Rows: client test sets. Columns: model. Values: mean ± SD across all site-split × fold pairs.
# Note that "Client i" refers to a positional index — patient identities differ per site split.

# In[74]:


rows = []
for i in range(N_CLIENTS):
    n_te_avg = int(np.mean([len(fold_y_tests[s][f][i]) for s, f in SF_PAIRS]))
    ev_avg   = np.mean([fold_y_tests[s][f][i][event_field].sum() for s, f in SF_PAIRS])
    row = {'Test Set': f'C{i+1}  (n≈{n_te_avg}, ev≈{ev_avg:.1f})'}
    for name in MODEL_NAMES_ALL:
        v = _model_folds(name, i)
        row[name] = f'{np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1):.4f}'
    rows.append(row)

mean_row = {'Test Set': 'Mean'}
for name in MODEL_NAMES_ALL:
    vals = np.concatenate([_model_folds(name, i) for i in range(N_CLIENTS)])
    mean_row[name] = f'{np.nanmean(vals):.4f} ± {np.nanstd(vals, ddof=1):.4f}'
rows.append(mean_row)

summary_df = pd.DataFrame(rows).set_index('Test Set')
print(f'C-index — mean ± SD over {N_SITE_SPLITS} site splits × {N_FOLDS} folds = {N_SF} runs')
print(summary_df.to_string())

_local_pooled = np.concatenate([_model_folds('Local', i)              for i in range(N_CLIENTS)])
_fedN_pooled  = np.concatenate([_model_folds(K_LABELS[N_CLIENTS], i)  for i in range(N_CLIENTS)])
_fair_pooled  = np.concatenate([_model_folds('Centralized-SRF', i)    for i in range(N_CLIENTS)])
_cent_pooled  = np.concatenate([_model_folds('Centralized', i)    for i in range(N_CLIENTS)])
_gap_fair     = np.nanmean(_fair_pooled) - np.nanmean(_local_pooled)
_gap_ceiling  = np.nanmean(_cent_pooled) - np.nanmean(_local_pooled)
_gap_realised = np.nanmean(_fedN_pooled) - np.nanmean(_local_pooled)
_recovered    = (_gap_realised / _gap_ceiling * 100) if abs(_gap_ceiling) > 1e-6 else float('nan')
print()
print('Baseline diagnostics (pooled means, ignoring undefined C-index runs):')
print(f'  Local               = {np.nanmean(_local_pooled):.4f}')
print(f'  Fed({N_CLIENTS})              = {np.nanmean(_fedN_pooled):.4f}  '
      f'(Δ vs Local = {_gap_realised:+.4f})')
print(f'  Centralized-SRF     = {np.nanmean(_fair_pooled):.4f}  '
      f'(Δ vs Local = {_gap_fair:+.4f})')
print(f'  Centralized     = {np.nanmean(_cent_pooled):.4f}  '
      f'(Δ vs Local = {_gap_ceiling:+.4f}  ← complete-feature oracle)')
print(f'  Federation recovered {_recovered:+.1f}% of the available Local→Centralized gap.')


# ## 7.5 Statistical Significance Tests
# 
# Paired Wilcoxon signed-rank test (non-parametric, two-sided) + paired *t*-test.  
# **Expected outcomes** are noted per comparison (α = 0.05):
# 
# | Comparison | Expected |
# |---|---|
# | Local → Fed (10 clients) | ✓ significant |
# | Fed (10 clients) → Centralized-SRF | ✗ not significant |
# | Local → Centralized-SRF | ✓ significant |
# 
# Each pair = one *(site\_split, fold)* evaluation on the **same test set**.

# In[93]:


from scipy.stats import wilcoxon, ttest_rel

local_name = K_LABELS[1]          # 'Local'
fed10_name = K_LABELS[N_CLIENTS]  # 'Fed (10 clients)'
fair_label = 'Centralized-SRF'    # display label

def _fair_folds(i):
    return np.array([fold_fair_c[s][f][i] for s, f in SF_PAIRS])

# ── pooled arrays ─────────────────────────────────────────────────────────────
local_arr = np.concatenate([_model_folds(local_name, i) for i in range(N_CLIENTS)])
fed10_arr = np.concatenate([_model_folds(fed10_name, i) for i in range(N_CLIENTS)])
fair_arr  = np.concatenate([_fair_folds(i)              for i in range(N_CLIENTS)])


def _sig_row(a, b, label_a, label_b, expected_sig):
    diff = b - a
    mask = ~(np.isnan(a) | np.isnan(b))
    a_, b_ = a[mask], b[mask]
    try:
        _, p_w = wilcoxon(a_, b_, alternative='two-sided')
    except ValueError:
        p_w = np.nan
    _, p_t = ttest_rel(a_, b_)
    sig = not np.isnan(p_w) and p_w < 0.05
    return {
        'Comparison'      : f'{label_a}  →  {label_b}',
        'N pairs'         : int(mask.sum()),
        'Mean Δ C-index'  : f"{np.nanmean(diff):+.4f}",
        'Median Δ C-index': f"{np.nanmedian(diff):+.4f}",
        'Wilcoxon p'      : f'{p_w:.4g}',
        'Paired-t p'      : f'{p_t:.4g}',
        'Observed'        : '✓ sig.' if sig else '✗ n.s.',
        'Expected'        : '✓ sig.' if expected_sig else '✗ n.s.',
        'Match?'        : '✓' if (sig == expected_sig) else '✗',
    }


# ── pooled summary ────────────────────────────────────────────────────────────
pooled_rows = [
    _sig_row(local_arr, fed10_arr, local_name, fed10_name, expected_sig=True),
    _sig_row(fed10_arr, fair_arr,  fed10_name, fair_label, expected_sig=False),
    _sig_row(local_arr, fair_arr,  local_name, fair_label, expected_sig=True),
]
pooled_sig_df = pd.DataFrame(pooled_rows).set_index('Comparison')
print('=== Pooled across all clients ===')
display(pooled_sig_df)


# ── per-client breakdown ──────────────────────────────────────────────────────
pc_rows = []
for i in range(N_CLIENTS):
    loc_i  = _model_folds(local_name, i)
    fed_i  = _model_folds(fed10_name, i)
    fair_i = _fair_folds(i)

    try:
        _, p_lf = wilcoxon(loc_i, fed_i,  alternative='two-sided', nan_policy='omit')
    except ValueError:
        p_lf = np.nan
    try:
        _, p_ff = wilcoxon(fed_i, fair_i, alternative='two-sided', nan_policy='omit')
    except ValueError:
        p_ff = np.nan
    try:
        _, p_lfair = wilcoxon(loc_i, fair_i, alternative='two-sided', nan_policy='omit')
    except ValueError:
        p_lfair = np.nan

    pc_rows.append({
        'Client'                  : f'C{i+1}',
        'N pairs'                 : int(np.sum(~(np.isnan(loc_i) | np.isnan(fed_i)))),
        'Mean Δ (Loc→Fed10)'      : f"{np.nanmean(fed_i  - loc_i):+.4f}",
        'p (Loc→Fed10)'           : f'{p_lf:.4g}',
        'Sig? [exp: ✓]'           : '✓' if (not np.isnan(p_lf)    and p_lf    < 0.05) else '✗',
        'Mean Δ (Fed10→Fair)'     : f"{np.nanmean(fair_i  - fed_i):+.4f}",
        'p (Fed10→Fair)'          : f'{p_ff:.4g}',
        'Sig? [exp: ✗]'           : '✓' if (not np.isnan(p_ff)    and p_ff    < 0.05) else '✗',
        'Mean Δ (Loc→Fair)'       : f"{np.nanmean(fair_i  - loc_i):+.4f}",
        'p (Loc→Fair)'            : f'{p_lfair:.4g}',
        'Sig? [exp: ✓] '          : '✓' if (not np.isnan(p_lfair) and p_lfair < 0.05) else '✗',
    })

pc_sig_df = pd.DataFrame(pc_rows).set_index('Client')
print('\n=== Per-client breakdown ===')
display(pc_sig_df)


# ## 8. Per-Site-Split Summary
# 
# How does pooled C-index vary across the random patient partitions? Each row is one site split.

# In[76]:


site_rows = []
for s in range(N_SITE_SPLITS):
    row = {'Site split': f'split{s+1} (seed={site_seeds[s]})'}
    for name in MODEL_NAMES_ALL:
        per_fold = np.array([_pooled_cindex(name, s, f) for f in range(N_FOLDS)])
        row[name] = f'{np.nanmean(per_fold):.4f}'
    site_rows.append(row)

site_summary = pd.DataFrame(site_rows).set_index('Site split')
print('Per-site-split pooled C-index (mean across CV folds, pooled across clients):')
print(site_summary.to_string())

print('\nPer-fold pooled C-index distribution (mean ± SD, 5–95 pct over all site×fold):')
print(f"{'Model':<22} {'Mean':>8}  {'SD':>7}  {'5–95 % CI':>20}")
print('─' * 60)
for name in MODEL_NAMES_ALL:
    pooled = np.array([_pooled_cindex(name, s, f) for s, f in SF_PAIRS])
    m, sd = np.nanmean(pooled), np.nanstd(pooled, ddof=1)
    lo, hi = np.nanpercentile(pooled, [5, 95])
    print(f'{name:<22} {m:>8.4f}  {sd:>7.4f}  [{lo:.4f}, {hi:.4f}]')


# ## 9. Long-format Results Export
# 
# One row per (site_split, fold, client, model) — ready for downstream stats packages.

# In[77]:


records = []
for s, f in SF_PAIRS:
    for i in range(N_CLIENTS):
        for name in MODEL_NAMES_ALL:
            if name == 'Centralized-SRF':
                ci = fold_fair_c[s][f][i]; risk = fold_fair_risk[s][f][i]; ntrees = np.nan
            elif name == 'Centralized':
                ci = fold_cent_c[s][f][i]; risk = fold_cent_risk[s][f][i]; ntrees = np.nan
            else:
                k = next(kk for kk, lbl in K_LABELS.items() if lbl == name)
                ci = fold_eval[s][f][k][i]['c_index']
                risk = fold_eval[s][f][k][i]['risk']
                ntrees = fold_eval[s][f][k][i]['n_fed_trees']
            metrics = risk_group_metrics(risk, fold_y_tests[s][f][i])
            records.append({
                'site_split': s,
                'site_seed' : site_seeds[s],
                'fold'      : f,
                'client'    : i + 1,
                'model'     : name,
                'k'         : next((kk for kk, lbl in K_LABELS.items() if lbl == name), np.nan),
                'c_index'   : ci,
                'rmst_diff' : metrics['rmst_diff'],
                'risk_short_diff': metrics['risk_short_diff'],
                'risk_med_diff'  : metrics['risk_med_diff'],
                'risk_long_diff' : metrics['risk_long_diff'],
                'n_fed_trees': ntrees,
                'train_n'   : len(fold_y_trains[s][f][i]),
                'test_n'    : len(fold_y_tests[s][f][i]),
                'test_events': int(fold_y_tests[s][f][i][event_field].sum()),
            })
results_df = pd.DataFrame(records)
out_csv = Path('federated_benchmark_BRCA_results.csv')
results_df.to_csv(out_csv, index=False)
print(f'Wrote {len(results_df)} rows to {out_csv.resolve()}')
results_df.head()


# ## 10. Convergence Plot — All Client Subset Combinations
# 
# For every non-empty subset `S` of clients we federate the members and evaluate each member on its own test set.
# Each (subset, client) point pools `N_SITE_SPLITS × N_FOLDS` values; the bold dot is the mean.
# 
# - `k=1` (Local): `N` subsets × 1 client each.
# - `k>1`: `C(N, k)` subsets × `k` evaluated clients.
# - Right-most column: complete-feature Centralized on each test set.
# 
# Dot color encodes which client's test set is being evaluated.

# In[78]:


subset_entries = []
for k in range(1, N_CLIENTS + 1):
    for subset in sorted(_combinations(range(N_CLIENTS), k)):
        fs    = frozenset(subset)
        label = '{' + ','.join(f'C{i+1}' for i in sorted(subset)) + '}'
        for i in subset:
            vals = np.array([fold_subset_pc[s][f][(fs, i)] for s, f in SF_PAIRS])
            subset_entries.append({'k': k, 'subset_label': label, 'client_idx': i, 'vals': vals})

n_entries = len(subset_entries)
CENT_X = N_CLIENTS + 0.80
rng_j = np.random.default_rng(42)
entry_jitter = rng_j.uniform(-0.32, 0.32, n_entries)
cent_jitter  = np.linspace(-0.20, 0.20, N_CLIENTS)

fig, ax = plt.subplots(figsize=(13, 6))
for idx, ent in enumerate(subset_entries):
    cli  = ent['client_idx']
    x0   = ent['k'] + entry_jitter[idx]
    vals = ent['vals']
    ax.scatter([x0] * len(vals), vals, color=CLIENT_COLORS[cli], marker=CLIENT_MARKERS[cli],
               s=18, edgecolors='none', alpha=0.18, zorder=4)
    ax.scatter(x0, np.nanmean(vals), color=CLIENT_COLORS[cli], marker=CLIENT_MARKERS[cli],
               s=80, edgecolors='#333333', linewidths=0.6, alpha=0.95, zorder=5)

for i in range(N_CLIENTS):
    cent_vals = np.array([fold_cent_c[s][f][i] for s, f in SF_PAIRS])
    x0 = CENT_X + cent_jitter[i]
    ax.scatter([x0] * len(cent_vals), cent_vals, color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=24, edgecolors='none', alpha=0.18, zorder=6)
    ax.scatter(x0, np.nanmean(cent_vals), color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=180, edgecolors=COLORS['Centralized'], linewidths=2.0, alpha=0.95, zorder=7)
    ax.annotate(f'C{i+1}', (x0, np.nanmean(cent_vals)), xytext=(0, 7),
                textcoords='offset points', ha='center', fontsize=7.5, fontweight='bold',
                color=COLORS['Centralized'])

ax.axhline(0.5, color='#aaaaaa', ls=':', lw=1.0)
for i in range(N_CLIENTS):
    ax.scatter([], [], color=CLIENT_COLORS[i], marker=CLIENT_MARKERS[i],
               s=80, edgecolors='#333333', linewidths=0.6, label=f'C{i+1} test set')
ax.add_artist(ax.legend(fontsize=9, loc='lower right', ncol=2, title='Test set (color)'))
cent_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                     markeredgecolor=COLORS['Centralized'], markeredgewidth=2.0, markersize=10,
                     label='Centralized\n(thick outline)')
ax.legend(handles=[cent_handle], fontsize=9, loc='upper right')

k_tick_pos = list(range(1, N_CLIENTS + 1)) + [CENT_X]
k_tick_names = []
for k in range(1, N_CLIENTS + 1):
    n_combos = len(list(_combinations(range(N_CLIENTS), k)))
    n_dots   = sum(1 for ent in subset_entries if ent['k'] == k)
    k_tick_names.append(f'{K_LABELS[k].replace(" (", chr(10)+"(")}\n({n_combos} combo{"s" if n_combos > 1 else ""}, {n_dots} dots)')
k_tick_names.append(f'Centralized\nRSF\n({N_CLIENTS} dots)')
ax.set_xticks(k_tick_pos)
ax.set_xticklabels(k_tick_names, fontsize=8.5)
ax.set_xlabel('Federation size k  ·  dot color = which client test set is evaluated')
ax.set_ylabel("C-index (evaluated on that client's own test set)")
ax.set_title(f'Federated Convergence — All Client Subset Combinations\n'
             f'Bold dot = mean over {N_SF} runs ({N_SITE_SPLITS} site splits × {N_FOLDS} folds)  ·  '
             f'faded dots = per-run values  ·  thick outline = Centralized',
             fontweight='bold')
plt.tight_layout()
plt.savefig('convergence.pdf', bbox_inches='tight')
plt.show()


# ## 11. C-index Boxplot
# 
# Left: per-client boxplots (one box per model, distribution over `N_SITE_SPLITS × N_FOLDS` runs).
# Right: pooled-across-clients boxplots.

# In[92]:


SHOW_FIG_A_LABELS = False   # mean/std on each per-client box (usually too busy)
SHOW_FIG_B_LABELS = True    # mean/std above each pooled box

SAVE_PDF      = True
PDF_FIG_A     = 'cindex_per_client.pdf'
PDF_FIG_B     = 'cindex_pooled.pdf'

# Publication-friendly defaults
plt.rcParams.update({
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  9,
    'legend.frameon':   True,
    'pdf.fonttype':     42,   # editable text in vector PDFs
    'ps.fonttype':      42,
})

# Sort models by pooled mean C-index (worst → best left-to-right)
BAR_MODELS = list(MODEL_NAMES_ALL)
_pooled_mean = {
    name: np.nanmean(np.concatenate(
        [_model_folds(name, i) for i in range(N_CLIENTS)]
    ))
    for name in BAR_MODELS
}
BAR_MODELS = sorted(BAR_MODELS, key=lambda n: _pooled_mean[n])


# ======================================================================
# Figure 1 — per-client C-index by model
# ======================================================================
fig1, ax = plt.subplots(figsize=(15, 6.5))

x_c     = np.arange(N_CLIENTS)
n_m     = len(BAR_MODELS)
w       = min(0.16, 0.88 / n_m)
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

for j, name in enumerate(BAR_MODELS):
    per_client_vals = [_model_folds(name, i) for i in range(N_CLIENTS)]
    positions       = x_c + offsets[j]

    bp = ax.boxplot(
        per_client_vals,
        positions=positions,
        widths=w * 0.85,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        whis=(0, 100),
        manage_ticks=False,
    )
    _style_box(bp, COLORS[name], HATCHES[name])

    rng_p = np.random.default_rng(j * 101 + 7)
    for pos, vals in zip(positions, per_client_vals):
        jit = rng_p.uniform(-w * 0.15, w * 0.15, len(vals))
        ax.scatter(
            pos + jit, vals,
            s=5, color='#111111', alpha=0.25, zorder=5,
        )

        if SHOW_FIG_A_LABELS:
            m  = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
            ax.text(
                pos, float(np.nanmax(vals)) + 0.012,
                f'{m:.2f}\n±{sd:.2f}',
                ha='center', va='bottom', fontsize=5.5,
            )

# Reference line at C-index = 0.5 (chance)
ax.axhline(0.5, color='#888888', ls=':', linewidth=1.0, label='_nolegend_')

# Light vertical separators between clients
for k in range(1, N_CLIENTS):
    ax.axvline(k - 0.5, color='#dddddd', linewidth=0.6, zorder=0)

ax.set_xticks(x_c)
ax.set_xticklabels([f'C{i+1}' for i in range(N_CLIENTS)])
ax.set_xlim(-0.5, N_CLIENTS - 0.5)
ax.set_xlabel('Client')
ax.set_ylabel('C-index on own test set')
ax.set_ylim(0.30, 1.05)
ax.grid(axis='y', alpha=0.35)
ax.set_title(
    f'Per-client C-index by model '
    f'({N_SITE_SPLITS} site splits × {N_FOLDS} folds = {N_SF} runs per box)'
)

# Legend below the plot, two rows — never overlaps the axes
legend_handles = [
    Patch(facecolor=COLORS[n], edgecolor='#111111', hatch=HATCHES[n], label=n)
    for n in BAR_MODELS
]
ncol = min(6, len(BAR_MODELS))
ax.legend(
    handles=legend_handles,
    ncol=ncol,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.11),
    fontsize=9,
    handlelength=2.0,
    columnspacing=1.4,
    borderaxespad=0.4,
)

plt.tight_layout()
if SAVE_PDF:
    plt.savefig(PDF_FIG_A, bbox_inches='tight', dpi=300)
plt.show()


# ======================================================================
# Figure 2 — pooled across clients (one box per model)
# ======================================================================
from matplotlib.lines import Line2D  # add near imports if you prefer

fig2, ax2 = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)

x_s = np.arange(len(BAR_MODELS))

# Precompute pooled values once
pooled_by_model = {
    name: np.concatenate([_model_folds(name, i) for i in range(N_CLIENTS)])
    for name in BAR_MODELS
}

# Reserve a clean annotation band above the actual data
data_max   = max(float(np.nanmax(v)) for v in pooled_by_model.values())
DATA_TOP   = min(1.02, max(0.965, data_max + 0.015))
LABEL_MEAN = DATA_TOP + 0.022
LABEL_STD  = DATA_TOP + 0.058
YLIM_TOP   = LABEL_STD + 0.040

for j, name in enumerate(BAR_MODELS):
    pooled_vals = pooled_by_model[name]

    bp = ax2.boxplot(
        [pooled_vals],
        positions=[j],
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        whis=(2.5, 97.5),
        showfliers=False,
        manage_ticks=False,
    )
    _style_box(bp, COLORS[name], HATCHES[name])

    # Keep all statistical lines visible above the points
    for key in ("boxes", "whiskers", "caps", "medians", "means"):
        for artist in bp.get(key, []):
            artist.set_clip_on(True)
            artist.set_zorder(7 if key in ("medians", "means") else 5)

    for artist in bp["medians"]:
        artist.set_color("#000000")
        artist.set_linewidth(1.25)

    for artist in bp["means"]:
        artist.set_color("#cc2222")
        artist.set_linewidth(1.6)
        artist.set_linestyle("-")

    rng_p = np.random.default_rng(j * 99 + 3)
    jit   = rng_p.uniform(-0.12, 0.12, len(pooled_vals))

    ax2.scatter(
        np.full(len(pooled_vals), j) + jit,
        pooled_vals,
        s=5,
        color="#111111",
        alpha=0.16,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )

    if SHOW_FIG_B_LABELS:
        m  = float(np.nanmean(pooled_vals))
        sd = (
            float(np.nanstd(pooled_vals, ddof=1))
            if len(pooled_vals) > 1 else np.nan
        )

        ax2.text(
            j, LABEL_MEAN, f"{m:.3f}",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
            clip_on=True,
        )
        ax2.text(
            j, LABEL_STD, f"±{sd:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#555555",
            clip_on=True,
        )

# Horizontal rule separating annotation band from data
ax2.axhline(DATA_TOP, color="#cccccc", linewidth=0.7, zorder=1)

# Reference line at C-index = 0.5
ax2.axhline(0.5, color="#888888", ls=":", linewidth=1.1, zorder=2)

ax2.set_xticks(x_s)
ax2.set_xticklabels(
    BAR_MODELS,
    fontsize=10,
    rotation=30,
    ha="right",
    rotation_mode="anchor",
)

ax2.set_xlim(-0.6, len(BAR_MODELS) - 0.4)
ax2.set_ylabel("C-index")
data_min = min(float(np.nanmin(v)) for v in pooled_by_model.values())
YLIM_BOT = min(0.30, data_min - 0.02)
ax2.set_ylim(YLIM_BOT, YLIM_TOP)

# Only tick the real C-index range; annotation band stays visually separate
ax2.set_yticks(np.arange(0.3, 1.01, 0.1))

ax2.grid(axis="y", alpha=0.25, linewidth=0.7)
ax2.set_axisbelow(True)

ax2.set_title(
    f"Pooled C-index across clients "
    f"({N_SF} × {N_CLIENTS} = {N_SF * N_CLIENTS} values per box)",
    pad=10,
)

# Legend for plot encodings, not models, since model names are already on x-axis
legend_handles = [
    Line2D([0], [0], color="#cc2222", lw=1.6, label="Mean"),
    Line2D([0], [0], color="#000000", lw=1.25, label="Median"),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor="#111111",
        markeredgecolor="none",
        alpha=0.25,
        markersize=5,
        label="Run value",
    ),
    Line2D([0], [0], color="#888888", lw=1.1, ls=":", label="Chance = 0.5"),
]

ax2.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=4,
    frameon=True,
    fontsize=9,
    handlelength=2.2,
    columnspacing=1.4,
)

if SAVE_PDF:
    fig2.savefig(PDF_FIG_B, bbox_inches="tight", dpi=300)

plt.show()


# ## 12. RMST Discrimination per Test Set × Model
# 
# Δ RMST = RMST(low-risk) − RMST(high-risk) in days, computed per (site, fold).
# Larger positive Δ = better separation of the median-risk-split groups.

# In[83]:


model_labels = MODEL_NAMES_ALL
xlbls = [_short_model_label(n) for n in model_labels]
x = np.arange(len(model_labels))

panel_specs = [(i, f'C{i+1} test') for i in range(N_CLIENTS)]
panel_stats = []
for panel_id, panel_title in panel_specs:
    vals_by_model = [_model_metric_folds('rmst_diff', name, panel_id) for name in model_labels]
    panel_stats.append((panel_id, panel_title, vals_by_model))

avg_vals_by_model = [
    np.concatenate([_model_metric_folds('rmst_diff', name, i) for i in range(N_CLIENTS)])
    for name in model_labels
]

all_finite = []
for _, _, vbm in panel_stats:
    for v in vbm:
        v = v[np.isfinite(v)]
        if v.size:
            all_finite.append(v)
for v in avg_vals_by_model:
    v = v[np.isfinite(v)]
    if v.size:
        all_finite.append(v)
all_vals = np.concatenate(all_finite) if all_finite else np.array([0.0])
y_span = max(1.0, float(np.max(all_vals) - np.min(all_vals)))
y_pad  = max(25.0, 0.12 * y_span)
text_offset = max(6.0, 0.025 * y_span)
y_min = min(0.0, float(np.min(all_vals)) - y_pad)
y_max = float(np.max(all_vals)) + y_pad + 2.0 * text_offset

fig, axes = plt.subplots(
    1, len(panel_specs) + 1,
    figsize=(4.6 * len(panel_specs) + 4.2, 6.4),
    sharey=True, constrained_layout=True,
    gridspec_kw={'width_ratios': [1.0] * len(panel_specs) + [0.9]},
)
axes = np.atleast_1d(axes)


def _draw_rmst_panel(ax, vals_by_model, panel_title, panel_seed, annotate_fs=7.0):
    for j, name in enumerate(model_labels):
        vals = vals_by_model[j]
        vals_f = vals[np.isfinite(vals)]
        bp_vals = vals_f if vals_f.size else np.array([0.0])
        bp = ax.boxplot([bp_vals], positions=[j], widths=0.62, patch_artist=True,
                        showmeans=True, meanline=True, whis=(0, 100), manage_ticks=False)
        _style_box(bp, COLORS[name], HATCHES[name])
        if vals_f.size:
            rng_p = np.random.default_rng(1000 * panel_seed + 101 * j + 7)
            jit = rng_p.uniform(-0.10, 0.10, len(vals_f))
            ax.scatter(np.full(len(vals_f), j) + jit, vals_f, s=8, color='#111111', alpha=0.45, zorder=5)
            m = float(np.nanmean(vals_f))
            sd = float(np.nanstd(vals_f, ddof=1)) if len(vals_f) > 1 else np.nan
            y_txt = float(np.max(vals_f)) + text_offset
            _annotate_mean_std(ax, j, y_txt, m, sd, '{:+.0f}', '+/-{:.0f}',
                               mean_fs=annotate_fs, std_fs=max(4.8, annotate_fs - 1.2))
    ax.axhline(0, color='#555555', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(xlbls, rotation=28, ha='right', fontsize=9.0)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.set_xlim(-0.70, len(model_labels) - 0.30); ax.set_ylim(y_min, y_max)
    ax.set_title(panel_title, fontsize=11.0, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10.0, fontweight='bold')


for p, (panel_id, panel_title, vbm) in enumerate(panel_stats):
    _draw_rmst_panel(axes[p], vbm, panel_title, p)
    if p == 0:
        axes[p].set_ylabel(f'Δ RMST = RMST(low) − RMST(high)  [days, horizon {RMST_HORIZON:.0f} d]',
                           fontsize=10.5, fontweight='bold')
_draw_rmst_panel(axes[-1], avg_vals_by_model, f'Average\n({N_SF} runs × {N_CLIENTS} clients)',
                 len(panel_stats) + 17, annotate_fs=8.5)
axes[-1].spines['left'].set_linewidth(1.4)
axes[-1].tick_params(axis='x', labelsize=8.5)
fig.suptitle(f'RMST Discrimination per Test Set × Model — {N_SITE_SPLITS} site splits × {N_FOLDS} folds',
             fontsize=13.0, fontweight='bold')
plt.savefig('rmst.pdf', bbox_inches='tight')
plt.show()


# ## 13. Risk-at-Time Discrimination
# 
# Rows = client test sets, columns = T₁ / T₂ / T₃.
# Δ Risk@T = Risk(high) − Risk(low) at the column horizon.

# In[84]:


horizons = [
    ('T₁ (short)', 'risk_short', T_SHORT),
    ('T₂ (med)',   'risk_med',   T_MED),
    ('T₃ (long)',  'risk_long',  T_LONG),
]

model_labels_rt = MODEL_NAMES_ALL
xlbls_rt = [_short_model_label(n) for n in model_labels_rt]
x_rt = np.arange(len(model_labels_rt))

n_rows_rt = N_CLIENTS
fig, axes = plt.subplots(n_rows_rt, 3, figsize=(15.0, 3.0 * n_rows_rt),
                         sharey='row', constrained_layout=True)
if n_rows_rt == 1:
    axes = np.array([axes])

row_specs = [(i, f'C{i+1} test') for i in range(N_CLIENTS)]
for row, (row_id, row_lbl) in enumerate(row_specs):
    col_stats = []
    row_vals = []
    for label, key, t_val in horizons:
        vals_by_model = [_model_metric_folds(f'{key}_diff', name, row_id) for name in model_labels_rt]
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
            bp = ax.boxplot([bp_vals], positions=[j], widths=0.62, patch_artist=True,
                            showmeans=True, meanline=True, whis=(0, 100), manage_ticks=False)
            _style_box(bp, COLORS[name], HATCHES[name])
            if vals_f.size:
                rng_p = np.random.default_rng(10000 * row + 1000 * col + 101 * j + 3)
                jit = rng_p.uniform(-0.10, 0.10, len(vals_f))
                ax.scatter(np.full(len(vals_f), j) + jit, vals_f, s=7, color='#111111', alpha=0.4, zorder=5)
                m = float(np.nanmean(vals_f))
                sd = float(np.nanstd(vals_f, ddof=1)) if len(vals_f) > 1 else np.nan
                y_txt = float(np.max(vals_f)) + y_txt_offset
                _annotate_mean_std(ax, j, y_txt, m, sd, '{:+.2f}', '+/-{:.2f}', mean_fs=6.5, std_fs=5.2)
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
            ax.set_ylabel(f'C{row_id+1}\nΔ P(T ≤ t)', fontsize=9.8, fontweight='bold')
        if row == 0:
            ax.set_title(f'{label}\nt = {t_val:.0f} d ({t_val/365:.1f} yr)',
                         fontsize=11.0, fontweight='bold')

fig.suptitle(f'Risk-at-Time Discrimination — Δ P(T ≤ t)  ·  {N_SITE_SPLITS} site splits × {N_FOLDS} folds',
             fontsize=13.0, fontweight='bold')
plt.savefig('risk_at_time.pdf', bbox_inches='tight')
plt.show()


# ## 14. Kaplan–Meier Curves
# 
# One panel per client test set. Bold curves = mean survival across all `(site_split, fold)` runs.
# Risk groups are defined by the median predicted risk *within each run*, computed from each model.

# In[85]:


KM_MODELS = [
    (1,             K_LABELS[1],         '-',                2.0, 0.85),
    (N_CLIENTS,     K_LABELS[N_CLIENTS], '--',               2.0, 0.85),
    ('fair',        'Centralized-SRF',   (0, (5, 1)),        2.0, 0.85),
    ('centralized', 'Centralized',   (0, (3, 1, 1, 1)),  2.0, 0.85),
]

fig, outer_axes = plt.subplots(2, N_CLIENTS, figsize=(5 * N_CLIENTS, 9),
                               gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08})
nar_times = np.array([0, T_SHORT, T_MED, T_LONG])
t_grid    = np.linspace(0.0, float(RMST_HORIZON), 200)


def _fold_risk(k_id, s, f, i):
    if k_id == 'centralized':
        return fold_cent_risk[s][f][i]
    if k_id == 'fair':
        return fold_fair_risk[s][f][i]
    return fold_eval[s][f][k_id][i]['risk']


def _km_on_grid(events, times, grid):
    if events.sum() < 1:
        return None
    km_t, km_s = kaplan_meier_estimator(events, times)
    if len(km_t) == 0:
        return None
    f_ = interp1d(km_t, km_s, kind='previous', bounds_error=False,
                  fill_value=(1.0, float(km_s[-1])))
    return f_(grid)


for col, i in enumerate(range(N_CLIENTS)):
    ax_km  = outer_axes[0, col]
    ax_nar = outer_axes[1, col]
    for k_id, model_name, ls, lw, alpha in KM_MODELS:
        for grp_name, color in [('Low', C_LOW), ('High', C_HIGH)]:
            curves_on_grid = []
            for s, f in SF_PAIRS:
                y_te   = fold_y_tests[s][f][i]
                events = y_te[event_field]
                times  = y_te[time_field]
                risk   = _fold_risk(k_id, s, f, i)
                mask   = (risk >= np.median(risk)) if grp_name == 'High' else ~(risk >= np.median(risk))
                e_, t_ = events[mask], times[mask]
                if e_.sum() < 1:
                    continue
                km_t, km_s = kaplan_meier_estimator(e_, t_)
                ax_km.step(km_t, km_s, where='post', color=color, lw=lw * 0.55,
                           ls=ls, alpha=0.12, zorder=3)
                s_grid = _km_on_grid(e_, t_, t_grid)
                if s_grid is not None:
                    curves_on_grid.append(s_grid)
            if curves_on_grid:
                mean_curve = np.mean(np.vstack(curves_on_grid), axis=0)
                lbl = (f'{model_name} — {grp_name} risk' if col == 0 else None)
                ax_km.plot(t_grid, mean_curve, color=color, lw=lw, ls=ls, alpha=alpha, label=lbl, zorder=5)
    c_loc  = _model_folds(K_LABELS[1], i)
    c_fed  = _model_folds(K_LABELS[N_CLIENTS], i)
    c_fair = _model_folds('Centralized-SRF', i)
    c_cent = _model_folds('Centralized', i)
    ann_lines = [
        f'{K_LABELS[1]}: C={np.nanmean(c_loc):.3f}±{np.nanstd(c_loc, ddof=1):.3f}',
        f'{K_LABELS[N_CLIENTS]}: C={np.nanmean(c_fed):.3f}±{np.nanstd(c_fed, ddof=1):.3f}',
        f'Centralized-SRF: C={np.nanmean(c_fair):.3f}±{np.nanstd(c_fair, ddof=1):.3f}',
        f'Centralized: C={np.nanmean(c_cent):.3f}±{np.nanstd(c_cent, ddof=1):.3f}',
    ]
    ax_km.text(0.97, 0.97, '\n'.join(ann_lines), transform=ax_km.transAxes, ha='right', va='top',
               fontsize=7.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='#bbbbbb', alpha=0.9))

    pvals = []
    for s, f in SF_PAIRS:
        try:
            y_te_f = fold_y_tests[s][f][i]
            risk_ref = fold_eval[s][f][N_CLIENTS][i]['risk']
            _, pv = compare_survival(y_te_f, (risk_ref >= np.median(risk_ref)).astype(int))
            pvals.append(pv)
        except Exception:
            pass
    if pvals:
        pv_mean = float(np.mean(pvals))
        stars = ('***' if pv_mean < 0.001 else '**' if pv_mean < 0.01 else '*' if pv_mean < 0.05 else 'ns')
        ax_km.text(0.97, 0.50, f'Log-rank p̄ = {pv_mean:.4f} {stars}',
                   transform=ax_km.transAxes, ha='right', va='top', fontsize=8.5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#bbbbbb', alpha=0.9))
    for t_mark, lbl in [(T_SHORT, 'T₁'), (T_MED, 'T₂'), (T_LONG, 'T₃')]:
        ax_km.axvline(t_mark, color='#888888', ls=':', lw=1.0)
        ax_km.text(t_mark, 1.03, lbl, ha='center', fontsize=8.5,
                   transform=ax_km.get_xaxis_transform(), color='#555555')
    ax_km.axvline(RMST_HORIZON, color='#555555', ls='--', lw=1.2, label='RMST horizon')
    n_te_avg = int(np.mean([len(fold_y_tests[s][f][i]) for s, f in SF_PAIRS]))
    ev_avg   = np.mean([fold_y_tests[s][f][i][event_field].sum() for s, f in SF_PAIRS])
    ax_km.set_title(f'C{i+1}  (n≈{n_te_avg}, ev≈{ev_avg:.1f})', fontweight='bold')
    ax_km.set_ylim(-0.02, 1.12)
    ax_km.set_xlim(0, float(RMST_HORIZON))
    ax_km.set_ylabel('Survival probability' if col == 0 else '')
    ax_km.tick_params(labelbottom=False)
    ax_nar.set_facecolor('white'); ax_nar.set_xlim(ax_km.get_xlim()); ax_nar.set_ylim(-0.5, 1.5)
    ax_nar.set_yticks([0, 1]); ax_nar.set_yticklabels(['High', 'Low'], fontsize=8.5)
    ax_nar.tick_params(left=False, bottom=False); ax_nar.grid(False)
    ax_nar.spines[['top', 'right', 'left']].set_visible(False)
    for row_pos, grp_name, color in [(0, 'High', C_HIGH), (1, 'Low', C_LOW)]:
        for t_q in nar_times:
            counts = []
            for s, f in SF_PAIRS:
                y_te_f   = fold_y_tests[s][f][i]
                times_f  = y_te_f[time_field]
                risk_f   = fold_eval[s][f][N_CLIENTS][i]['risk']
                high_f   = risk_f >= np.median(risk_f)
                mask_f   = high_f if grp_name == 'High' else ~high_f
                counts.append(int(np.sum(times_f[mask_f] >= t_q)))
            ax_nar.text(t_q, row_pos, f'{np.mean(counts):.0f}', ha='center', va='center',
                        fontsize=8.5, color=color)
    ax_nar.set_xlabel('Time (days)')
    if col == 0:
        ax_nar.text(-0.22, 0.5, f'n at risk\n(Fed, avg over {N_SF} runs)',
                    transform=ax_nar.transAxes, fontsize=8, va='center', ha='right', style='italic')

legend_handles = (
    [Line2D([0], [0], color='#555555', ls=ls, lw=1.8, label=name)
     for _, name, ls, _, _ in KM_MODELS] +
    [Line2D([0], [0], color=C_HIGH, lw=2.0, label='High-risk'),
     Line2D([0], [0], color=C_LOW,  lw=2.0, label='Low-risk')]
)
outer_axes[0, 0].legend(handles=legend_handles, fontsize=8.5, loc='lower left')
fig.suptitle(f'Kaplan–Meier — Local vs Fed({N_CLIENTS}) vs Centralized-SRF vs Centralized\n'
             f'Bold = mean over {N_SF} runs ({N_SITE_SPLITS} site splits × {N_FOLDS} folds)  ·  '
             f'faded = per-run steps  ·  Color = risk group',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('km_curves.pdf', bbox_inches='tight')
plt.show()


# In[ ]:




