import os, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, ElasticNet as SklearnElasticNet
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    brier_score_loss, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
EPS = 1e-12


# =============================================================================
# Public release configuration
# Manuscript main analyses only.
# Use command-line arguments to point to your local data files.
# =============================================================================
DIARY_CSV: Optional[str] = None
VOCAB_CSV: Optional[str] = None
SURVEY_CSV: Optional[str] = None
OUT_DIR: Optional[Path] = None

SEED              = 91
OUTER_SPLITS      = 3
OUTER_REPEATS     = 20
INNER_SPLITS_THR  = 3
THR_MODE          = "youden"
SENS_TARGET       = 0.80
DAY_AGG           = "median"
PRT_AGG           = "mean"
ADD_ACOUSTIC_STD  = True
ADD_LEXICAL_STD   = False
RUN_TEMPORAL      = True
TEMP_MIN_DAYS_PER_PRT = 1
TEMP_WINDOW_SIZES = [3, 5, 7]
TEMP_EARLY_SPLIT_DAY  = 8
MAX_TEMP_DAY      = 15
DPI               = 400
NESTED_STABLE_EVAL = True
STABLE_MIN_RATE = 0.95
STABLE_FALLBACK_MIN_RATE = 0.80
STABLE_MIN_FEATURES = 3
STABLE_SELECTION_SPLITS = 3
STABLE_SELECTION_REPEATS = 20
ENABLE_GRID_SEARCH = False
GRID_SEARCH_INNER_SPLITS = 3
GRID_SEARCH_N_JOBS = -1

MODEL_WHITELIST = ["ElasticNet", "SVM(linear)", "RandomForest", "SVM(rbf)", "CNN"]

# =============================================================================
# Dataclasses
# =============================================================================
@dataclass
class Paths:
    diary_csv: str
    vocab_csv: str
    survey_csv: str

@dataclass
class EvalConfig:
    phq_healthy_max_exclusive: int = 4
    phq_depress_min_exclusive: int = 10
    outer_splits: int = 3
    outer_repeats: int = 20
    inner_splits_thr: int = 3
    threshold_mode: str = "youden"
    sens_target: float = 0.80
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    day_agg: str = "median"
    prt_agg: str = "mean"
    add_acoustic_std: bool = True
    add_lexical_std: bool = False
    phq9_col: Optional[str] = None
    shiftwork_col: Optional[str] = "shift_work"
    make_main_plots: bool = True
    make_supp_plots: bool = True
    make_bootstrap_plots: bool = True
    n_bootstrap_plot: int = 2000
    seed: int = 71
    dpi: int = 300
    nested_stable_eval: bool = True
    stable_min_rate: float = 0.95
    stable_fallback_min_rate: float = 0.80
    stable_min_features: int = 3
    stable_selection_splits: int = 3
    stable_selection_repeats: int = 20
    enable_grid_search: bool = False
    grid_search_inner_splits: int = 3
    grid_search_n_jobs: int = -1

# =============================================================================
# Feature definitions
# =============================================================================
ACOUSTIC_FEATURES_BASE = [
    "f0_mean", "f0_std", "pitch_slope_st_per_s", "hnr_mean_db", "voiced_frac",
    "rms_mean", "rms_std", "spec_centroid_mean", "spec_bandwidth_mean",
    "spec_flux_mean", "zcr_mean",
]

LEXICAL_CATEGORIES = {
    "pos_affect":      ["pos_high", "pos_calm"],
    "wellness":        ["exercise", "rest", "leisure", "mindfulness_religion"],
    "neg_dep":         ["neg_flat"],
    "neg_stress":      ["neg_tense", "conflict"],
    "health_concern":  ["symptom", "fatigue", "care"],
    "daily_burden":    ["chores", "admin", "economy"],
    "self_focus":      ["reflection", "improvement"],
}

LEXICAL_COLS_BASE = [
    "lex_neg_dep",
    "lex_neg_stress",
    "lex_daily_burden",
    "lex_wellness",
    "lex_pos_affect",
    "lex_self_focus",
    "domain_entropy",
    "global_sub_entropy",
    "lag1_word_jaccard_mean",
]

LEXICAL_HELPER_COLS = [
    "_sub_counts_json",
    "_word_list_json",
]

MODEL_ORDER = ["ElasticNet", "SVM(linear)", "RandomForest", "SVM(rbf)", "CNN"]
MODEL_COLORS = {
    "ElasticNet":   "#2ca02c",
    "SVM(linear)":  "#9467bd",
    "RandomForest": "#ff7f0e",
    "SVM(rbf)":     "#1f77b4",
    "CNN":          "#17becf",
}

MODEL_ALIASES = {
    "Elastic Net": "ElasticNet",
    "SVM_linear":  "SVM(linear)",
    "SVM_rbf":     "SVM(rbf)",
    "RF":          "RandomForest",
    "RandomForestClassifier": "RandomForest",
}

MODEL_PARAM_GRIDS_CLF = {
    "ElasticNet": {
        "m__C": [0.1, 1.0, 3.0],
        "m__l1_ratio": [0.2, 0.5, 0.8],
    },
    "SVM(linear)": {
        "m__C": [0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "m__max_depth": [4, 6, None],
        "m__min_samples_leaf": [1, 2, 4],
    },
    "SVM(rbf)": {
        "m__C": [0.3, 1.0, 3.0],
        "m__gamma": ["scale", 0.1, 0.01],
    },
}

MODEL_PARAM_GRIDS_REG = {
    "ElasticNet": {
        "m__alpha": [0.01, 0.1, 1.0],
        "m__l1_ratio": [0.2, 0.5, 0.8],
    },
    "SVM(linear)": {
        "m__C": [0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "m__max_depth": [4, 6, None],
        "m__min_samples_leaf": [1, 2, 4],
    },
    "SVM(rbf)": {
        "m__C": [0.3, 1.0, 3.0],
        "m__gamma": ["scale", 0.1, 0.01],
    },
}

# feature set 표시용 이름/스타일
DEMOGRAPHIC_BASELINE_KEY = "Demographic_baseline"

FEATURESET_DISPLAY_NAMES = {
    DEMOGRAPHIC_BASELINE_KEY: "Demographic baseline",
    "Acoustic_mean+std": "Acoustic features",
    "Lexical_mean": "Lexical features",
    "Combined_main": "Combined features",
    "Combined_stable": "Stability-selected reduced model",
}

FEATURESET_COLORS = {
    DEMOGRAPHIC_BASELINE_KEY: "#7f7f7f",
    "Acoustic": "#4C9AFF",
    "Lexical": "#37C493",
    "Combined_main": "#F24B65",
    "Combined_stable": "#4A4CC6FF",
}

COMPARISON_COLORS = {
    "Combined_main": "#F24B65",
    "Combined_stable": "#4A4CC6FF",
}

COMPARISON_FILL_COLORS = {
    "Combined_main": "#FAC8CCFF",
    "Combined_stable": "#D0D6F6FF",
}

MAIN_FEATURESET_PLOT_ORDER = [
    DEMOGRAPHIC_BASELINE_KEY,
    "Acoustic_mean+std",
    "Lexical_mean",
    "Combined_main",
]

SUPPLEMENTARY_FEATURESET_PLOT_ORDER = [
    *MAIN_FEATURESET_PLOT_ORDER,
    "Combined_stable",
]

# =============================================================================
# Small utilities (unchanged)
# =============================================================================
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_header(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)

def safe_auc(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    return float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")

def std_ddof0(x):
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return 0.0 if len(arr) <= 1 else float(np.std(arr, ddof=0))

def ece_quantile(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob).astype(float), 0.0, 1.0)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(prob, qs)
    edges[0], edges[-1] = 0.0, 1.0
    ece = 0.0
    n = len(prob)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        cnt = int(idx.sum())
        if cnt == 0:
            continue
        ece += (cnt / (n + EPS)) * abs(prob[idx].mean() - y_true[idx].mean())
    return float(ece)

def choose_threshold(prob, y_true, mode="youden", sens_target=0.80):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    thr_grid = np.unique(prob)
    if len(thr_grid) > 400:
        thr_grid = np.quantile(prob, np.linspace(0, 1, 401))
    best_t, best_score = 0.5, -np.inf
    for t in thr_grid:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + EPS)
        spec = tn / (tn + fp + EPS)
        if mode == "youden":
            score = sens + spec - 1
        elif mode == "sens80":
            score = spec if sens >= sens_target else -np.inf
        elif mode == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t if np.isfinite(best_score) else 0.5

def cluster_bootstrap_samples(metric_func, y_true, y_pred, groups, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    scores = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        try:
            value = metric_func(np.asarray(y_true)[idx], np.asarray(y_pred)[idx])
            if np.isfinite(value):
                scores.append(float(value))
        except Exception:
            continue
    return np.asarray(scores, dtype=float)

def cluster_bootstrap_ci(metric_func, y_true, y_pred, groups, n_bootstrap=1000, ci=0.95, seed=42):
    scores = cluster_bootstrap_samples(metric_func, y_true, y_pred, groups, n_bootstrap, seed)
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(scores.mean()), float(lo), float(hi)

def normalize_model_name(name):
    return MODEL_ALIASES.get(str(name).strip(), str(name).strip())

def normalize_model_dict_keys(models):
    out = {}
    for name, model in models.items():
        norm_name = normalize_model_name(name)
        if norm_name in out:
            raise ValueError(f"Duplicate model key: {name} -> {norm_name}")
        out[norm_name] = model
    return out

def normalize_model_whitelist(model_whitelist):
    if not model_whitelist:
        return None
    return {normalize_model_name(x) for x in model_whitelist}

def get_model_color(name):
    return MODEL_COLORS.get(normalize_model_name(name), "#333333")

def sort_models_for_plot(model_names):
    def _key(name):
        norm = normalize_model_name(name)
        return (0, MODEL_ORDER.index(norm), norm) if norm in MODEL_ORDER else (1, 999, norm)
    return sorted(model_names, key=_key)

def display_featureset_name(fs_name):
    return FEATURESET_DISPLAY_NAMES.get(fs_name, str(fs_name).replace("_", " "))

def featureset_category(fs_name):
    if fs_name == DEMOGRAPHIC_BASELINE_KEY:
        return DEMOGRAPHIC_BASELINE_KEY
    if fs_name.startswith("Acoustic"):
        return "Acoustic"
    if fs_name.startswith("Lexical"):
        return "Lexical"
    if fs_name == "Combined_main":
        return "Combined_main"
    if fs_name == "Combined_stable":
        return "Combined_stable"
    return "Combined_main"

def get_featureset_bar_style(fs_name):
    cat = featureset_category(fs_name)
    style = {
        "facecolor": FEATURESET_COLORS.get(cat, "#333333"),
        "edgecolor": "black",
        "linewidth": 1.0,
        "linestyle": "-",
        "hatch": None,
        "alpha": 0.92,
        "text_color": "white",
    }
    if fs_name == "Combined_stable":
        style.update({
            "facecolor": "white",
            "edgecolor": FEATURESET_COLORS["Combined_stable"],
            "linewidth": 1.8,
            "linestyle": "--",
            "hatch": "///",
            "alpha": 1.0,
            "text_color": FEATURESET_COLORS["Combined_stable"],
        })
    return style

def infer_nfeat_col(df):
    for col in ["n_features", "n_feats", "num_features"]:
        if col in df.columns:
            return col
    return None

def as_binary_score_1d(x):
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.astype(float).ravel()
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float).ravel()
        elif arr.shape[1] == 2:
            return arr[:, 1].astype(float).ravel()
    raise ValueError(f"Expected binary scores, got shape {arr.shape}")


def apply_zoomed_auroc_axis(ax, lo_arr, hi_arr, tick_step=0.05, min_floor=0.50, top_pad=0.03):
    """Set y-axis limits to zoom into the AUROC range for bar charts."""
    lo_arr = np.asarray(lo_arr, dtype=float)
    hi_arr = np.asarray(hi_arr, dtype=float)
    lo_arr = lo_arr[np.isfinite(lo_arr)]
    hi_arr = hi_arr[np.isfinite(hi_arr)]
    if len(lo_arr) == 0 or len(hi_arr) == 0:
        ax.set_ylim(0.0, 1.0)
        return 0.0, 1.0
    data_lo = float(np.min(lo_arr))
    data_hi = float(np.max(hi_arr))
    y_min = max(min_floor, np.floor(data_lo / tick_step) * tick_step - tick_step)
    y_max = min(1.0, np.ceil(data_hi / tick_step) * tick_step + top_pad)
    if y_max - y_min < 0.15:
        y_min = max(0.0, y_max - 0.20)
    ax.set_ylim(y_min, y_max)
    ticks = np.arange(y_min, y_max + tick_step / 2, tick_step)
    ax.set_yticks(ticks)
    return y_min, y_max


def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD) between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return float("nan")
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def auroc_diff_bootstrap_ci(y_true, prob_a, prob_b, prt_ids=None,
                             n_bootstrap=2000, ci=0.95, seed=42):
    """
    Bootstrap CI for AUROC(prob_a) - AUROC(prob_b).
    Cluster bootstrap if prt_ids provided.
    Returns: (delta, ci_lo, ci_hi)
    """
    rng = np.random.RandomState(seed)
    observed_delta = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    deltas = []
    if prt_ids is not None:
        unique_ids = np.unique(prt_ids)
        for _ in range(n_bootstrap):
            boot_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
            idx = np.concatenate([np.where(prt_ids == uid)[0] for uid in boot_ids])
            y_b = y_true[idx]
            if len(np.unique(y_b)) < 2:
                continue
            deltas.append(roc_auc_score(y_b, prob_a[idx]) - roc_auc_score(y_b, prob_b[idx]))
    else:
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            y_b = y_true[idx]
            if len(np.unique(y_b)) < 2:
                continue
            deltas.append(roc_auc_score(y_b, prob_a[idx]) - roc_auc_score(y_b, prob_b[idx]))
    deltas = np.array(deltas)
    alpha = 1.0 - ci
    return observed_delta, np.percentile(deltas, 100 * alpha / 2), np.percentile(deltas, 100 * (1 - alpha / 2))


# =============================================================================
# Torch CNN
# =============================================================================
class _CNN1DNet(nn.Module):
    def __init__(self, n_features, out_dim=1, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

class _BaseTorchCNN(BaseEstimator):
    def __init__(self, task, epochs, batch_size, lr, weight_decay, dropout,
                 random_state, device, val_frac, patience):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.device = device
        self.val_frac = val_frac
        self.patience = patience

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_random_state(self):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_loss(self, y_train, device):
        if self.task == "classification":
            y_int = y_train.astype(int)
            n_pos = max(1, int((y_int == 1).sum()))
            n_neg = max(1, int((y_int == 0).sum()))
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.MSELoss()

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

        n_val = max(1, int(len(y) * self.val_frac))
        X_val, y_val = X[:n_val], y[:n_val]
        X_tr, y_tr = X[n_val:], y[n_val:]

        device = self._get_device()
        self.device_ = str(device)
        self.n_features_in_ = X.shape[1]
        self.model_ = _CNN1DNet(self.n_features_in_, out_dim=1, dropout=self.dropout).to(device)

        criterion = self._build_loss(y_tr, device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size, shuffle=False,
        )

        self._set_random_state()

        best_val, best_state, bad_epochs = float("inf"), None, 0
        for _ in range(int(self.epochs)):
            self.model_.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device).view(-1, 1)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_losses = [
                    float(criterion(self.model_(xb.to(device)), yb.to(device).view(-1, 1)).item())
                    for xb, yb in val_loader
                ]
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device_)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(X).to(device)).view(-1).detach().cpu().numpy().astype(float)

class TorchCNNClassifier(_BaseTorchCNN, ClassifierMixin):
    def __init__(self, epochs=120, batch_size=32, lr=1e-3, weight_decay=1e-4,
                 dropout=0.25, random_state=42, device=None, val_frac=0.15, patience=15):
        super().__init__("classification", epochs, batch_size, lr, weight_decay,
                         dropout, random_state, device, val_frac, patience)

    def predict_proba(self, X):
        logits = self._raw_predict(X)
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1.0 - prob, prob]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class TorchCNNRegressor(_BaseTorchCNN, RegressorMixin):
    def __init__(self, epochs=160, batch_size=32, lr=1e-3, weight_decay=1e-4,
                 dropout=0.25, random_state=42, device=None, val_frac=0.15, patience=15):
        super().__init__("regression", epochs, batch_size, lr, weight_decay,
                         dropout, random_state, device, val_frac, patience)

    def predict(self, X):
        return self._raw_predict(X)

# =============================================================================
# Model registry — SVM(rbf) GridSearchCV 제거
# =============================================================================
def build_models(seed=42) -> Tuple[Dict[str, Pipeline], Dict[str, Pipeline]]:
    clf_models = {
        "ElasticNet": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)),
        ]),
        "SVM(linear)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", SVC(kernel="linear", C=1.0, probability=True,
                      class_weight="balanced", random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestClassifier(n_estimators=400, max_depth=6,
                                          class_weight="balanced", random_state=seed)),
        ]),
        # [변경] GridSearchCV 제거 → 고정 HP
        "SVM(rbf)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                      class_weight="balanced", random_state=seed)),
        ]),
        "CNN": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", TorchCNNClassifier(epochs=120, batch_size=32, lr=1e-3,
                                      weight_decay=1e-4, dropout=0.25, random_state=seed)),
        ]),
    }

    reg_models = {
        "ElasticNet": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", SklearnElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed)),
        ]),
        "SVM(linear)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", SVR(kernel="linear", C=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(n_estimators=400, max_depth=6, random_state=seed)),
        ]),
        # [변경] GridSearchCV 제거
        "SVM(rbf)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", SVR(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "CNN": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", TorchCNNRegressor(epochs=160, batch_size=32, lr=1e-3,
                                     weight_decay=1e-4, dropout=0.25, random_state=seed)),
        ]),
    }
    return normalize_model_dict_keys(clf_models), normalize_model_dict_keys(reg_models)

# =============================================================================
# Data building (unchanged)
# =============================================================================
def standardize_prt_day_columns(df):
    out = df.copy()
    if "prt" not in out.columns:
        raise KeyError("Missing 'prt' column.")
    out["prt"] = out["prt"].astype(str).str.strip()
    if "day" in out.columns:
        out["day"] = pd.to_numeric(out["day"], errors="coerce")
    return out

def build_target_table(df_survey, cfg):
    phq_col = cfg.phq9_col or "phq9_1st"
    if phq_col not in df_survey.columns:
        raise KeyError(f"Missing column: {phq_col}")

    base_cols = ["prt", phq_col]
    optional_map = {"pss_1st": "pss", "gad7_1st": "gad7", "gender": "gender", "age": "age"}
    for src in optional_map:
        if src in df_survey.columns:
            base_cols.append(src)
    if cfg.shiftwork_col and cfg.shiftwork_col in df_survey.columns:
        base_cols.append(cfg.shiftwork_col)

    df_target = df_survey[base_cols].copy()
    rename_map = {phq_col: "phq9", **{src: dst for src, dst in optional_map.items() if src in df_target.columns}}
    df_target = df_target.rename(columns=rename_map)
    df_target["phq9"] = pd.to_numeric(df_target["phq9"], errors="coerce")
    df_target = df_target.dropna(subset=["phq9"]).copy()

    df_target["group"] = pd.NA
    df_target.loc[df_target["phq9"] <= cfg.phq_healthy_max_exclusive, "group"] = "healthy"
    df_target.loc[df_target["phq9"] >= cfg.phq_depress_min_exclusive, "group"] = "depression"
    df_target = df_target[df_target["group"].notna()].copy()
    df_target["group"] = df_target["group"].astype(str).str.strip().str.lower()
    return df_target

def build_day_acoustic(df_diary, cfg):
    df_diary = df_diary.copy()
    mfcc_cols = [c for c in [f"mfcc{i}_mean" for i in range(1, 14)] if c in df_diary.columns]
    acoustic_cols = [c for c in ACOUSTIC_FEATURES_BASE if c in df_diary.columns]

    for col in sorted(set(mfcc_cols) | set(acoustic_cols)):
        df_diary[col] = pd.to_numeric(df_diary[col], errors="coerce")

    if len(mfcc_cols) >= 3:
        df_diary["mfcc_mean_avg"] = df_diary[mfcc_cols].mean(axis=1)
        df_diary["mfcc_mean_std"] = df_diary[mfcc_cols].std(axis=1, ddof=0)
        acoustic_cols += ["mfcc_mean_avg", "mfcc_mean_std"]

    if not acoustic_cols:
        raise RuntimeError("No acoustic features found.")

    df_diary = df_diary.dropna(subset=["day"]).copy()
    df_diary["day"] = df_diary["day"].astype(int)
    agg = "median" if cfg.day_agg == "median" else "mean"
    df_day_ac = df_diary.groupby(["prt", "day"], as_index=False)[acoustic_cols].agg(agg)
    return df_day_ac, acoustic_cols

def build_day_lexical(df_vocab):
    df_vocab = df_vocab.dropna(subset=["day", "word", "main_domain", "sub_domain"]).copy()
    df_vocab["day"] = df_vocab["day"].astype(int)
    df_vocab["word"] = df_vocab["word"].astype(str).str.strip()

    def _entropy_from_series(s):
        probs = s.value_counts(normalize=True).to_numpy(dtype=float)
        return float(-(probs * np.log(probs + EPS)).sum()) if len(probs) else 0.0

    day_rows = []
    for (prt, day), g in df_vocab.groupby(["prt", "day"]):
        sub_counts = g["sub_domain"].value_counts()
        row = {"prt": prt, "day": int(day)}
        for category, subs in LEXICAL_CATEGORIES.items():
            row[f"lex_{category}"] = float(sum(sub_counts.get(s, 0) for s in subs))
        row["domain_entropy"] = _entropy_from_series(g["main_domain"])
        row["_sub_counts_json"] = json.dumps(g["sub_domain"].value_counts().to_dict(), ensure_ascii=False, sort_keys=True)
        row["_word_list_json"] = json.dumps(g["word"].tolist(), ensure_ascii=False)
        day_rows.append(row)

    df_day_lex = pd.DataFrame(day_rows)
    if len(df_day_lex) == 0:
        raise RuntimeError("No lexical rows built.")
    for col in LEXICAL_COLS_BASE:
        if col not in df_day_lex.columns:
            df_day_lex[col] = 0.0
    return df_day_lex, list(LEXICAL_COLS_BASE)

def load_and_build_day_level(paths, cfg):
    print_header("DATA LOADING & DAY-LEVEL BUILD")

    df_diary = standardize_prt_day_columns(pd.read_csv(paths.diary_csv))
    df_vocab = standardize_prt_day_columns(pd.read_csv(paths.vocab_csv))
    df_survey = standardize_prt_day_columns(pd.read_csv(paths.survey_csv))

    valid_prt = set(df_survey["prt"]) & set(df_diary["prt"]) & set(df_vocab["prt"])
    df_diary = df_diary[df_diary["prt"].isin(valid_prt)].copy()
    df_vocab = df_vocab[df_vocab["prt"].isin(valid_prt)].copy()
    df_survey = df_survey[df_survey["prt"].isin(valid_prt)].copy()

    print(f"[LOAD] diary={df_diary.shape}, vocab={df_vocab.shape}, survey={df_survey.shape}")
    print(f"[FREEZE] complete-case intersection N={len(valid_prt)}")

    df_target = build_target_table(df_survey, cfg)
    print(f"[TARGET] N={df_target['prt'].nunique()} | {df_target['group'].value_counts().to_dict()}")

    df_day_ac, acoustic_cols = build_day_acoustic(df_diary, cfg)
    acoustic_feature_count = len(acoustic_cols) * (2 if cfg.add_acoustic_std else 1)
    print(
        f"[ACOUSTIC] {acoustic_feature_count} participant-level features "
        f"({len(acoustic_cols)} day-level + {len(acoustic_cols) if cfg.add_acoustic_std else 0} std) | "
        f"rows={len(df_day_ac)} | day_agg={cfg.day_agg}"
    )

    df_day_lex, lexical_cols = build_day_lexical(df_vocab)
    print(f"[LEXICAL] {len(lexical_cols)} features | rows={len(df_day_lex)}")

    df_main = (
        df_day_ac
        .merge(df_day_lex, on=["prt", "day"], how="inner")
        .merge(df_target, on="prt", how="inner")
    ).copy()
    df_main = df_main.dropna(subset=["prt", "day", "phq9", "group"]).copy()
    df_main["day"] = df_main["day"].astype(int)
    df_main["group"] = df_main["group"].astype(str).str.strip().str.lower()
    print(f"[DATA] rows={len(df_main)} | prt={df_main['prt'].nunique()} | days={df_main['day'].nunique()}")

    return df_main, acoustic_cols, lexical_cols

# =============================================================================
# Participant aggregation
# =============================================================================
def aggregate_participants(df_main, acoustic_cols, lexical_cols, cfg,
                           day_range=None, min_days_per_prt=1,
                           fixed_prt_ids=None):
    df = df_main.copy()
    df["group"] = df["group"].astype(str).str.strip().str.lower()

    if fixed_prt_ids is not None:
        df = df[df["prt"].astype(str).isin(set(map(str, fixed_prt_ids)))].copy()
    if day_range is not None:
        df = df[(df["day"] >= day_range[0]) & (df["day"] <= day_range[1])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    counts = df.groupby("prt")["day"].nunique()
    df = df[df["prt"].isin(counts[counts >= min_days_per_prt].index)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    meta_cols = ["phq9", "group", "age", "gender", "pss", "gad7"]
    if cfg.shiftwork_col and cfg.shiftwork_col in df.columns:
        meta_cols.append(cfg.shiftwork_col)

    lexical_daily_cols = [
        "lex_neg_dep",
        "lex_neg_stress",
        "lex_daily_burden",
        "lex_wellness",
        "lex_pos_affect",
        "lex_self_focus",
        "domain_entropy",
    ]

    for col in acoustic_cols + lexical_daily_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_dict = {col: cfg.prt_agg for col in acoustic_cols if col in df.columns}
    for col in meta_cols:
        if col in df.columns:
            agg_dict[col] = "first"

    df_prt = df.groupby("prt", as_index=False).agg(agg_dict)

    if set(LEXICAL_HELPER_COLS).issubset(df.columns):
        def _entropy_from_counts(counts):
            total = float(sum(counts.values()))
            if total <= 0:
                return 0.0
            probs = np.asarray(list(counts.values()), dtype=float) / total
            return float(-(probs * np.log(probs + EPS)).sum())

        def _jaccard(a, b):
            union = len(a | b)
            return float(len(a & b) / union) if union else 0.0

        lexical_rows = []
        for prt, g in df.groupby("prt"):
            g = g.sort_values("day").copy()
            day_lex = g[[col for col in lexical_daily_cols if col in g.columns]].agg(cfg.prt_agg)
            sub_counts_total = {}
            words_by_day = []

            for _, row in g.iterrows():
                sub_counts = json.loads(row.get("_sub_counts_json") or "{}")
                words_day = json.loads(row.get("_word_list_json") or "[]")

                for key, value in sub_counts.items():
                    sub_counts_total[key] = sub_counts_total.get(key, 0) + int(value)

                words_by_day.append([str(w).strip() for w in words_day if str(w).strip()])

            lag1_word_jaccard = []
            day_word_sets = [set(words_day) for words_day in words_by_day]
            for prev_set, curr_set in zip(day_word_sets[:-1], day_word_sets[1:]):
                lag1_word_jaccard.append(_jaccard(prev_set, curr_set))

            row_out = {
                "prt": prt,
                "global_sub_entropy": _entropy_from_counts(sub_counts_total),
                "lag1_word_jaccard_mean": float(np.mean(lag1_word_jaccard)) if lag1_word_jaccard else 0.0,
            }
            for col in lexical_daily_cols:
                if col in day_lex.index:
                    row_out[col] = float(day_lex[col])
            lexical_rows.append(row_out)

        df_lex = pd.DataFrame(lexical_rows)
        df_prt = df_prt.merge(df_lex, on="prt", how="left")
    else:
        for col in lexical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        lex_agg = {col: cfg.prt_agg for col in lexical_cols if col in df.columns}
        if lex_agg:
            df_lex = df.groupby("prt", as_index=False).agg(lex_agg)
            df_prt = df_prt.merge(df_lex, on="prt", how="left")

    if cfg.add_acoustic_std and acoustic_cols:
        std_ac = df.groupby("prt")[acoustic_cols].agg(std_ddof0).add_prefix("std_").reset_index()
        df_prt = df_prt.merge(std_ac, on="prt", how="left")

    return df_prt

# =============================================================================
# Feature sets — 자동 추출
# =============================================================================
def build_feature_sets(acoustic_cols, lexical_cols, cfg, stable_features=None):
    """
    stable_features가 None이면 3개 set (Acoustic, Lexical, Combined_main)만 반환.
    stable_features가 리스트면 Combined_stable도 추가.
    """
    acoustic_std = [f"std_{col}" for col in acoustic_cols]
    feature_sets = {
        "Acoustic_mean+std": acoustic_cols + acoustic_std,
        "Lexical_mean": lexical_cols,
        "Combined_main": acoustic_cols + acoustic_std + lexical_cols,
    }
    if stable_features is not None:
        feature_sets["Combined_stable"] = stable_features
    return feature_sets

def filter_existing_features(df, features):
    return [col for col in features if col in df.columns]

# =============================================================================
# Stable feature 추출
# =============================================================================
def get_stable_selection_cv(cfg):
    return int(cfg.stable_selection_splits), int(cfg.stable_selection_repeats)

def summarize_stable_features_from_matrix(
    X, y, feature_names, clf_pipeline,
    n_splits=3, n_repeats=20, seed=42,
    min_rate=0.95, fallback_min_rate=0.80, min_features=3,
    verbose=True,
):
    """Repeated CV on the provided matrix only; safe to call inside each outer-train split."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    sel_rows = []
    coef_rows = []
    for fold_id, (tr, te) in enumerate(rskf.split(X, y)):
        model = clone(clf_pipeline)
        model.fit(X[tr], y[tr])
        lr = model.named_steps.get("m", model.steps[-1][1])
        if hasattr(lr, "coef_"):
            coef = lr.coef_.ravel()
            sel_rows.append({"fold": fold_id, **{f: int(abs(c) > 1e-8) for f, c in zip(feature_names, coef)}})
            coef_rows.append({"fold": fold_id, **{f: abs(c) for f, c in zip(feature_names, coef)}})

    if not sel_rows:
        raise RuntimeError("No coefficients extracted during stable feature selection.")

    sel_matrix = pd.DataFrame(sel_rows).set_index("fold")
    coef_matrix = pd.DataFrame(coef_rows).set_index("fold")
    jaccard = compute_pairwise_jaccard(sel_matrix)
    selection_rate = sel_matrix.mean(axis=0)
    mean_abs_coef = coef_matrix.mean(axis=0)

    stable_mask = selection_rate >= min_rate
    if stable_mask.sum() < min_features:
        stable_mask = selection_rate >= fallback_min_rate
    stable_rates = selection_rate[stable_mask]
    stable_coefs = mean_abs_coef[stable_mask]
    sort_key = pd.DataFrame({"rate": stable_rates, "coef": stable_coefs})
    sort_key = sort_key.sort_values(["rate", "coef"], ascending=[False, False])
    stable = sort_key.index.tolist()

    if verbose:
        print(f"  Pairwise Jaccard = {jaccard:.3f}")
        always_selected = selection_rate[selection_rate >= 1.0].index.tolist()
        print(f"  Features selected in ALL folds: {len(always_selected)}")
        print(
            f"\n  Stability-selected reduced-model features "
            f"(rate >= {min_rate:.0%} or fallback {fallback_min_rate:.0%}): {len(stable)}"
        )
        for i, f in enumerate(stable, 1):
            rate = selection_rate[f]
            coef = mean_abs_coef[f]
            tag = " *" if rate >= 1.0 else ""
            print(f"    {i}. {f:<30s}  rate={rate:.1%}  |c|={coef:.4f}{tag}")

    return {
        "stable_features": stable,
        "jaccard": float(jaccard),
        "sel_matrix": sel_matrix,
        "selection_rate": selection_rate.sort_values(ascending=False),
        "mean_abs_coef": mean_abs_coef.sort_values(ascending=False),
    }

def compute_fold_selection_matrix(X, y, feature_names, clf_pipeline,
                                  outer_splits=3, outer_repeats=20, seed=42):
    """각 outer fold에서 ElasticNet이 선택한 feature를 0/1 matrix로 반환."""
    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    rows = []
    for fold_id, (tr, te) in enumerate(rskf.split(X, y)):
        model = clone(clf_pipeline)
        model.fit(X[tr], y[tr])
        lr = model.named_steps.get("m", model.steps[-1][1])
        if hasattr(lr, "coef_"):
            coef = lr.coef_.ravel()
            selected = (np.abs(coef) > 1e-8).astype(int)
            row = {"fold": fold_id}
            for fname, sel in zip(feature_names, selected):
                row[fname] = sel
            rows.append(row)
    return pd.DataFrame(rows).set_index("fold")

def compute_pairwise_jaccard(matrix):
    arr = matrix.values.astype(bool)
    n = len(arr)
    if n < 2:
        return float("nan")
    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            union = np.logical_or(arr[i], arr[j]).sum()
            if union == 0:
                jaccards.append(1.0)
            else:
                inter = np.logical_and(arr[i], arr[j]).sum()
                jaccards.append(inter / union)
    return float(np.mean(jaccards))

def extract_stable_features(df_prt, combined_features, clf_pipeline, cfg,
                            min_rate=None):
    """
    [Method 3] Combined_main에서 ElasticNet을 반복 학습하여 stable feature를 자동 추출.

    selection_rate >= min_rate 인 feature만 선택.
    - target_k 없음 (개수를 강제하지 않음)
    - |coef| 랭킹으로 자르지 않음
    - 순수하게 "몇 % fold에서 선택되었는가"만으로 결정
    - 결과 feature 수는 데이터에 의해 결정됨
    """
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].reset_index(drop=True)
    features = filter_existing_features(df_hd, combined_features)
    X = df_hd[features].to_numpy()
    y = (df_hd["group"].values == "depression").astype(int)

    print_header("REDUCED-MODEL FEATURE SELECTION")
    if min_rate is None:
        min_rate = cfg.stable_min_rate
    stable_splits, stable_repeats = get_stable_selection_cv(cfg)
    result = summarize_stable_features_from_matrix(
        X, y, features, clf_pipeline,
        n_splits=stable_splits,
        n_repeats=stable_repeats,
        seed=cfg.seed,
        min_rate=min_rate,
        fallback_min_rate=cfg.stable_fallback_min_rate,
        min_features=cfg.stable_min_features,
        verbose=True,
    )
    return result["stable_features"], result["jaccard"], result["sel_matrix"]

# =============================================================================
# OOF helpers
# =============================================================================
def maybe_tune_estimator(X_train, y_train, estimator, model_name, task, cfg, fold_seed):
    """Optional GridSearchCV confined to the current outer-train split."""
    if (cfg is None) or (not cfg.enable_grid_search) or (not model_name):
        return clone(estimator), None

    model_name = normalize_model_name(model_name)
    grids = MODEL_PARAM_GRIDS_CLF if task == "clf" else MODEL_PARAM_GRIDS_REG
    param_grid = grids.get(model_name)
    if not param_grid:
        return clone(estimator), None

    if task == "clf":
        y_int = np.asarray(y_train).astype(int)
        class_counts = np.bincount(y_int)
        if len(class_counts) < 2:
            return clone(estimator), None
        max_splits = int(class_counts.min())
        if max_splits < 2:
            return clone(estimator), None
        n_splits = min(int(cfg.grid_search_inner_splits), max_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
        scoring = "roc_auc"
    else:
        n_splits = min(int(cfg.grid_search_inner_splits), len(y_train))
        if n_splits < 2:
            return clone(estimator), None
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
        scoring = "neg_mean_absolute_error"

    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=cfg.grid_search_n_jobs,
        error_score=np.nan,
    )
    try:
        search.fit(X_train, y_train)
    except Exception as exc:
        return clone(estimator), {"status": "failed", "error": str(exc)}

    tuned = clone(estimator).set_params(**search.best_params_)
    return tuned, {
        "status": "ok",
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "task": task,
    }

def make_outer_folds(y, cfg):
    splitter = RepeatedStratifiedKFold(n_splits=cfg.outer_splits, n_repeats=cfg.outer_repeats, random_state=cfg.seed)
    return list(splitter.split(np.zeros((len(y), 1)), y))

def inner_oof_probabilities(X_train, y_train, clf_pipeline, inner_splits, seed):
    skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    p_oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in skf.split(X_train, y_train):
        model = clone(clf_pipeline)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        p_oof[val_idx] = model.predict_proba(X_train[val_idx])[:, 1]
    return p_oof

def repeated_classifier_oof(X, y, clf, outer_folds, cfg=None, model_name=None):
    prob_sum = np.zeros(len(y), dtype=float)
    prob_cnt = np.zeros(len(y), dtype=float)
    vote_sum = np.zeros(len(y), dtype=float) if cfg is not None else None
    vote_cnt = np.zeros(len(y), dtype=float) if cfg is not None else None
    thr_list = []
    search_info = []

    for fold_id, (tr_idx, te_idx) in enumerate(outer_folds):
        model, gs_info = maybe_tune_estimator(
            X[tr_idx], y[tr_idx], clf, model_name, task="clf",
            cfg=cfg, fold_seed=cfg.seed + 4000 + fold_id,
        ) if cfg is not None else (clone(clf), None)
        model.fit(X[tr_idx], y[tr_idx])
        p_te = model.predict_proba(X[te_idx])[:, 1]
        prob_sum[te_idx] += p_te
        prob_cnt[te_idx] += 1.0
        if gs_info:
            search_info.append({"fold": fold_id, **gs_info})
        if cfg is not None:
            p_tr_oof = inner_oof_probabilities(X[tr_idx], y[tr_idx], model, cfg.inner_splits_thr, cfg.seed + 2000 + fold_id)
            thr = choose_threshold(p_tr_oof, y[tr_idx], mode=cfg.threshold_mode, sens_target=cfg.sens_target)
            thr_list.append(float(thr))
            vote_sum[te_idx] += (p_te >= thr).astype(int)
            vote_cnt[te_idx] += 1.0

    out = {"oof_prob": prob_sum / np.clip(prob_cnt, 1.0, None), "counts": prob_cnt, "thr_list": thr_list}
    if cfg is not None:
        out["vote_rate"] = vote_sum / np.clip(vote_cnt, 1.0, None)
    if search_info:
        out["search_info"] = search_info
    return out

def repeated_regressor_oof(X, y, reg, outer_folds, cfg=None, model_name=None):
    pred_sum = np.zeros(len(y), dtype=float)
    pred_cnt = np.zeros(len(y), dtype=float)
    search_info = []
    for fold_id, (tr_idx, te_idx) in enumerate(outer_folds):
        try:
            model, gs_info = maybe_tune_estimator(
                X[tr_idx], y[tr_idx], reg, model_name, task="reg",
                cfg=cfg, fold_seed=cfg.seed + 6000 + fold_id,
            ) if cfg is not None else (clone(reg), None)
            model.fit(X[tr_idx], y[tr_idx])
            pred_sum[te_idx] += model.predict(X[te_idx])
            pred_cnt[te_idx] += 1.0
            if gs_info:
                search_info.append({"fold": fold_id, **gs_info})
        except Exception:
            continue
    return pred_sum / np.clip(pred_cnt, 1.0, None), search_info

def resolve_global_threshold(oof_prob, X, y, clf, cfg, thr_list, policy="median_fold_thr"):
    if policy == "median_fold_thr":
        thr = float(np.median(thr_list)) if thr_list else 0.5
        return thr, "median(fold thresholds)"
    elif policy == "mean_fold_thr":
        thr = float(np.mean(thr_list)) if thr_list else 0.5
        return thr, "mean(fold thresholds)"
    raise ValueError(f"Unknown policy: {policy}")

# =============================================================================
# Metrics & evaluation (unchanged)
# =============================================================================
def compute_classification_metrics(y, prob, pred):
    auroc = safe_auc(y, prob)
    pr_auc = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, prob))
    ece = ece_quantile(y, prob, n_bins=10)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn + EPS)   # sensitivity = recall of positive class
    spec = tn / (tn + fp + EPS)   # specificity
    ppv  = tp / (tp + fp + EPS)   # precision / PPV
    npv  = tn / (tn + fn + EPS)
    acc  = (tp + tn) / (tp + tn + fp + fn + EPS)
    f1   = 2 * ppv * sens / (ppv + sens + EPS)
    bal_acc = (sens + spec) / 2.0

    return {
        "AUROC": auroc,
        "PR_AUC": pr_auc,
        "Brier": brier,
        "ECE_q": ece,
        "Acc": float(acc),
        "Sens": float(sens),
        "Spec": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "BalancedAcc": float(bal_acc),
        "F1": float(f1),
    }

def compute_regression_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {"Reg_r": np.nan, "Reg_rho": np.nan, "MAE": np.nan, "mask": mask}
    r, _ = pearsonr(y_true[mask], y_pred[mask])
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    return {"Reg_r": float(r), "Reg_rho": float(rho), "MAE": float(mae), "mask": mask}

def evaluate_hd_oof(df_prt, features, clf, reg, cfg, compute_bootstrap_ci_flag=True, model_name=None):
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].reset_index(drop=True)
    n_h = int((df_hd["group"] == "healthy").sum())
    n_d = int((df_hd["group"] == "depression").sum())
    if n_h == 0 or n_d == 0:
        raise RuntimeError(f"Need both groups. H={n_h} D={n_d}")

    features = filter_existing_features(df_hd, features)
    X = df_hd[features].to_numpy()
    y = (df_hd["group"].values == "depression").astype(int)
    y_reg = pd.to_numeric(df_hd["phq9"], errors="coerce").to_numpy(dtype=float)
    prt_ids = df_hd["prt"].values

    outer_folds = make_outer_folds(y, cfg)
    clf_oof = repeated_classifier_oof(X, y, clf, outer_folds, cfg=cfg, model_name=model_name)
    reg_oof, reg_search_info = repeated_regressor_oof(X, y_reg, reg, outer_folds, cfg=cfg, model_name=model_name)

    oof_prob = clf_oof["oof_prob"]
    thr_list = clf_oof["thr_list"]
    thr_global, thr_note = resolve_global_threshold(oof_prob, X, y, clf, cfg, thr_list)
    strict_preds = (oof_prob >= thr_global).astype(int)

    cls_metrics = compute_classification_metrics(y, oof_prob, strict_preds)
    cls_report, cm = compute_binary_detailed_report(y, strict_preds, oof_prob)
    reg_metrics = compute_regression_metrics(y_reg, reg_oof)
    reg_mask = reg_metrics.pop("mask")

    if compute_bootstrap_ci_flag:
        auroc_ci = cluster_bootstrap_ci(
            lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
            y, oof_prob, prt_ids, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed,
        )
        if reg_mask.sum() >= 3:
            reg_r_ci = cluster_bootstrap_ci(
                lambda a, b: pearsonr(a, b)[0] if len(a) >= 3 else float("nan"),
                y_reg[reg_mask], reg_oof[reg_mask], prt_ids[reg_mask],
                n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed + 123,
            )
        else:
            reg_r_ci = (float(reg_metrics["Reg_r"]), float("nan"), float("nan"))
    else:
        auroc_ci = (float(cls_metrics["AUROC"]), float("nan"), float("nan"))
        reg_r_ci = (float(reg_metrics["Reg_r"]), float("nan"), float("nan"))

    metrics = {
        **cls_metrics,
        "AUROC_lo": auroc_ci[1],
        "AUROC_hi": auroc_ci[2],
        **reg_metrics,
        "Reg_r_lo": reg_r_ci[1],
        "Reg_r_hi": reg_r_ci[2],
    }
    return {
        "oof_prob": oof_prob,
        "oof_reg": reg_oof,
        "y": y,
        "y_reg": y_reg,
        "prt_ids": prt_ids,
        "strict_preds": strict_preds,
        "confusion_matrix": cm,              
        "classification_report": cls_report, 
        "thresholds": {
            "mode": cfg.threshold_mode,
            "thr_global": float(thr_global),
            "thr_mean": float(np.mean(thr_list)) if thr_list else float("nan"),
            "thr_std": float(np.std(thr_list)) if thr_list else float("nan"),
            "thr_list_fold": [float(x) for x in thr_list],
            "note": thr_note,
        },
        "metrics": metrics,
        "search_info": {
            "clf": clf_oof.get("search_info", []),
            "reg": reg_search_info,
        },
        "n_features_report": len(features),
    }

def evaluate_hd_oof_nested_stable(
    df_prt, combined_features, clf, reg, selector_clf, cfg,
    compute_bootstrap_ci_flag=True, model_name=None,
):
    """Evaluate Combined_stable without leakage by re-selecting features inside each outer-train split."""
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].reset_index(drop=True)
    n_h = int((df_hd["group"] == "healthy").sum())
    n_d = int((df_hd["group"] == "depression").sum())
    if n_h == 0 or n_d == 0:
        raise RuntimeError(f"Need both groups. H={n_h} D={n_d}")

    base_features = filter_existing_features(df_hd, combined_features)
    X_full = df_hd[base_features].to_numpy()
    y = (df_hd["group"].values == "depression").astype(int)
    y_reg = pd.to_numeric(df_hd["phq9"], errors="coerce").to_numpy(dtype=float)
    prt_ids = df_hd["prt"].values

    outer_folds = make_outer_folds(y, cfg)
    feature_to_idx = {f: i for i, f in enumerate(base_features)}
    prob_sum = np.zeros(len(y), dtype=float)
    prob_cnt = np.zeros(len(y), dtype=float)
    reg_sum = np.zeros(len(y_reg), dtype=float)
    reg_cnt = np.zeros(len(y_reg), dtype=float)
    thr_list = []
    outer_sel_rows = []
    nested_search = {"clf": [], "reg": []}

    stable_splits, stable_repeats = get_stable_selection_cv(cfg)
    for fold_id, (tr_idx, te_idx) in enumerate(outer_folds):
        stable_result = summarize_stable_features_from_matrix(
            X_full[tr_idx], y[tr_idx], base_features, selector_clf,
            n_splits=stable_splits,
            n_repeats=stable_repeats,
            seed=cfg.seed + 10000 + fold_id,
            min_rate=cfg.stable_min_rate,
            fallback_min_rate=cfg.stable_fallback_min_rate,
            min_features=cfg.stable_min_features,
            verbose=False,
        )
        selected_features = stable_result["stable_features"]
        if len(selected_features) < cfg.stable_min_features:
            raise RuntimeError(
                f"Stable selection returned too few features in fold {fold_id}: {len(selected_features)}"
            )

        sel_idx = [feature_to_idx[f] for f in selected_features]
        X_train = X_full[tr_idx][:, sel_idx]
        X_test = X_full[te_idx][:, sel_idx]

        clf_model, clf_search = maybe_tune_estimator(
            X_train, y[tr_idx], clf, model_name, task="clf",
            cfg=cfg, fold_seed=cfg.seed + 11000 + fold_id,
        )
        clf_model.fit(X_train, y[tr_idx])
        p_te = clf_model.predict_proba(X_test)[:, 1]
        prob_sum[te_idx] += p_te
        prob_cnt[te_idx] += 1.0

        p_tr_oof = inner_oof_probabilities(
            X_train, y[tr_idx], clf_model, cfg.inner_splits_thr, cfg.seed + 12000 + fold_id
        )
        thr = choose_threshold(
            p_tr_oof, y[tr_idx],
            mode=cfg.threshold_mode,
            sens_target=cfg.sens_target,
        )
        thr_list.append(float(thr))

        try:
            reg_model, reg_search = maybe_tune_estimator(
                X_train, y_reg[tr_idx], reg, model_name, task="reg",
                cfg=cfg, fold_seed=cfg.seed + 13000 + fold_id,
            )
            reg_model.fit(X_train, y_reg[tr_idx])
            reg_sum[te_idx] += reg_model.predict(X_test)
            reg_cnt[te_idx] += 1.0
            if reg_search:
                nested_search["reg"].append({"fold": fold_id, **reg_search})
        except Exception:
            pass

        if clf_search:
            nested_search["clf"].append({"fold": fold_id, **clf_search})
        outer_sel_rows.append({
            "fold": fold_id,
            "n_features": len(selected_features),
            **{f: int(f in selected_features) for f in base_features},
        })

    oof_prob = prob_sum / np.clip(prob_cnt, 1.0, None)
    reg_oof = reg_sum / np.clip(reg_cnt, 1.0, None)
    thr_global, thr_note = resolve_global_threshold(oof_prob, X_full, y, clf, cfg, thr_list)
    strict_preds = (oof_prob >= thr_global).astype(int)

    cls_metrics = compute_classification_metrics(y, oof_prob, strict_preds)
    cls_report, cm = compute_binary_detailed_report(y, strict_preds, oof_prob)
    reg_metrics = compute_regression_metrics(y_reg, reg_oof)
    reg_mask = reg_metrics.pop("mask")

    if compute_bootstrap_ci_flag:
        auroc_ci = cluster_bootstrap_ci(
            lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
            y, oof_prob, prt_ids, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed,
        )
        if reg_mask.sum() >= 3:
            reg_r_ci = cluster_bootstrap_ci(
                lambda a, b: pearsonr(a, b)[0] if len(a) >= 3 else float("nan"),
                y_reg[reg_mask], reg_oof[reg_mask], prt_ids[reg_mask],
                n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed + 123,
            )
        else:
            reg_r_ci = (float(reg_metrics["Reg_r"]), float("nan"), float("nan"))
    else:
        auroc_ci = (float(cls_metrics["AUROC"]), float("nan"), float("nan"))
        reg_r_ci = (float(reg_metrics["Reg_r"]), float("nan"), float("nan"))

    outer_sel_df = pd.DataFrame(outer_sel_rows).set_index("fold")
    n_feature_counts = outer_sel_df["n_features"].to_numpy(dtype=float)
    selection_frequency = outer_sel_df.drop(columns=["n_features"]).mean(axis=0).sort_values(ascending=False)

    metrics = {
        **cls_metrics,
        "AUROC_lo": auroc_ci[1],
        "AUROC_hi": auroc_ci[2],
        **reg_metrics,
        "Reg_r_lo": reg_r_ci[1],
        "Reg_r_hi": reg_r_ci[2],
    }
    return {
        "oof_prob": oof_prob,
        "oof_reg": reg_oof,
        "y": y,
        "y_reg": y_reg,
        "prt_ids": prt_ids,
        "strict_preds": strict_preds,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "thresholds": {
            "mode": cfg.threshold_mode,
            "thr_global": float(thr_global),
            "thr_mean": float(np.mean(thr_list)) if thr_list else float("nan"),
            "thr_std": float(np.std(thr_list)) if thr_list else float("nan"),
            "thr_list_fold": [float(x) for x in thr_list],
            "note": thr_note,
        },
        "metrics": metrics,
        "search_info": nested_search,
        "n_features_report": int(np.median(n_feature_counts)) if len(n_feature_counts) else 0,
        "n_features_min": int(np.min(n_feature_counts)) if len(n_feature_counts) else 0,
        "n_features_max": int(np.max(n_feature_counts)) if len(n_feature_counts) else 0,
        "nested_stable": {
            "selection_matrix": outer_sel_df.drop(columns=["n_features"]),
            "selection_frequency": selection_frequency.to_dict(),
            "n_features_mean": float(np.mean(n_feature_counts)) if len(n_feature_counts) else float("nan"),
            "n_features_median": float(np.median(n_feature_counts)) if len(n_feature_counts) else float("nan"),
        },
    }

def compute_binary_detailed_report(y_true, pred, prob=None):
    """
    Positive class = depression (1)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(pred).astype(int).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    ppv  = tp / (tp + fp + EPS)
    npv  = tn / (tn + fn + EPS)
    acc  = (tp + tn) / (tp + tn + fp + fn + EPS)
    f1   = 2 * ppv * sens / (ppv + sens + EPS)
    bal_acc = (sens + spec) / 2.0

    report = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Precision_PPV": float(ppv),
        "NPV": float(npv),
        "Balanced_Accuracy": float(bal_acc),
        "F1": float(f1),
    }

    if prob is not None:
        prob = np.asarray(prob).astype(float).ravel()
        report["AUROC"] = safe_auc(y_true, prob)
        report["PR_AUC"] = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
        report["Brier"] = float(brier_score_loss(y_true, prob))
        report["ECE_q"] = ece_quantile(y_true, prob, n_bins=10)

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)
    return report, cm


def print_binary_report(report, title=None, round_digits=3):
    if title:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)
    rep_df = pd.DataFrame.from_dict(report, orient="index", columns=["value"])
    print(rep_df.round(round_digits).to_string())


def plot_confusion_matrix_heatmap(cm, title=None,
                                  labels=("Healthy", "Depression"),
                                  normalize=False, save_path=None, dpi=DPI,
                                  show_f1=True, positive_index=1, f1_digits=3,
                                  ax=None):
    cm = np.asarray(cm, dtype=float)
    cm_raw = cm.copy()

    if normalize:
        cm_plot = cm_raw / np.clip(cm_raw.sum(axis=1, keepdims=True), EPS, None)
        fmt = ".2f"
    else:
        cm_plot = cm_raw
        fmt = ".0f"

    f1_text = None
    if show_f1 and cm_raw.shape == (2, 2):
        if positive_index == 1:
            tp = cm_raw[1, 1]
            fp = cm_raw[0, 1]
            fn = cm_raw[1, 0]
        else:
            tp = cm_raw[0, 0]
            fp = cm_raw[1, 0]
            fn = cm_raw[0, 1]

        f1 = 2 * tp / np.clip(2 * tp + fp + fn, EPS, None)
        f1_text = f"F1={f1:.{f1_digits}f}"

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        created_fig = True
    else:
        fig = ax.figure

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
        yticklabels=[f"True {labels[0]}", f"True {labels[1]}"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )

    if title:
        ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    if f1_text:
        ax.text(0.5, -0.17, f1_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=13, fontweight="bold")

    if created_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()

    return fig, ax

def evaluate_temporal_point(df_prt, features, clf, cfg):
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].reset_index(drop=True)
    features = filter_existing_features(df_hd, features)
    n_h = int((df_hd["group"] == "healthy").sum())
    n_d = int((df_hd["group"] == "depression").sum())

    if len(features) < 3 or len(df_hd) < 20 or n_h == 0 or n_d == 0:
        return {
            "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
            "p_h": np.nan, "p_d": np.nan, "ordered": False,
            "n_hd": len(df_hd), "n_h": n_h, "n_d": n_d,
        }

    X = df_hd[features].to_numpy()
    y = (df_hd["group"].values == "depression").astype(int)
    prt_ids = df_hd["prt"].values
    grp = df_hd["group"].values

    outer_folds = make_outer_folds(y, cfg)
    oof_prob = repeated_classifier_oof(X, y, clf, outer_folds, cfg=None)["oof_prob"]
    auroc = safe_auc(y, oof_prob)
    _, auroc_lo, auroc_hi = cluster_bootstrap_ci(
        lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
        y, oof_prob, prt_ids, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed,
    )
    p_h = float(oof_prob[grp == "healthy"].mean()) if n_h > 0 else np.nan
    p_d = float(oof_prob[grp == "depression"].mean()) if n_d > 0 else np.nan
    ordered = bool(p_h < p_d) if np.isfinite(p_h) and np.isfinite(p_d) else False
    return {
        "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "p_h": p_h, "p_d": p_d, "ordered": ordered,
        "n_hd": len(df_hd), "n_h": n_h, "n_d": n_d,
    }

def evaluate_temporal_point_nested_stable(df_prt, combined_features, selector_clf, clf, cfg, model_name="ElasticNet"):
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].reset_index(drop=True)
    base_features = filter_existing_features(df_hd, combined_features)
    n_h = int((df_hd["group"] == "healthy").sum())
    n_d = int((df_hd["group"] == "depression").sum())

    if len(base_features) < cfg.stable_min_features or len(df_hd) < 20 or n_h == 0 or n_d == 0:
        return {
            "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
            "p_h": np.nan, "p_d": np.nan, "ordered": False,
            "n_hd": len(df_hd), "n_h": n_h, "n_d": n_d,
        }

    X_full = df_hd[base_features].to_numpy()
    y = (df_hd["group"].values == "depression").astype(int)
    prt_ids = df_hd["prt"].values
    grp = df_hd["group"].values
    feature_to_idx = {f: i for i, f in enumerate(base_features)}
    outer_folds = make_outer_folds(y, cfg)
    stable_splits, stable_repeats = get_stable_selection_cv(cfg)

    prob_sum = np.zeros(len(y), dtype=float)
    prob_cnt = np.zeros(len(y), dtype=float)

    for fold_id, (tr_idx, te_idx) in enumerate(outer_folds):
        stable_result = summarize_stable_features_from_matrix(
            X_full[tr_idx], y[tr_idx], base_features, selector_clf,
            n_splits=stable_splits,
            n_repeats=stable_repeats,
            seed=cfg.seed + 20000 + fold_id,
            min_rate=cfg.stable_min_rate,
            fallback_min_rate=cfg.stable_fallback_min_rate,
            min_features=cfg.stable_min_features,
            verbose=False,
        )
        selected_features = stable_result["stable_features"]
        if len(selected_features) < cfg.stable_min_features:
            continue

        sel_idx = [feature_to_idx[f] for f in selected_features]
        X_train = X_full[tr_idx][:, sel_idx]
        X_test = X_full[te_idx][:, sel_idx]
        clf_model, _ = maybe_tune_estimator(
            X_train, y[tr_idx], clf, model_name, task="clf",
            cfg=cfg, fold_seed=cfg.seed + 21000 + fold_id,
        )
        clf_model.fit(X_train, y[tr_idx])
        prob_sum[te_idx] += clf_model.predict_proba(X_test)[:, 1]
        prob_cnt[te_idx] += 1.0

    oof_prob = prob_sum / np.clip(prob_cnt, 1.0, None)
    auroc = safe_auc(y, oof_prob)
    _, auroc_lo, auroc_hi = cluster_bootstrap_ci(
        lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
        y, oof_prob, prt_ids, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci_level, seed=cfg.seed,
    )
    p_h = float(oof_prob[grp == "healthy"].mean()) if n_h > 0 else np.nan
    p_d = float(oof_prob[grp == "depression"].mean()) if n_d > 0 else np.nan
    ordered = bool(p_h < p_d) if np.isfinite(p_h) and np.isfinite(p_d) else False
    return {
        "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "p_h": p_h, "p_d": p_d, "ordered": ordered,
        "n_hd": len(df_hd), "n_h": n_h, "n_d": n_d,
    }

# =============================================================================
# Plotting functions
# =============================================================================
   
def plot_roc_all_models_for_featureset(y_true, oof_prob_by_model, panel_label, ax=None):
    y_true = np.asarray(y_true).astype(int).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 3.8))

    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, alpha=0.8, color="tab:blue", label="Chance")

    for model_name in sort_models_for_plot(list(oof_prob_by_model.keys())):
        prob = as_binary_score_1d(oof_prob_by_model[model_name])
        if len(prob) != len(y_true):
            continue

        auc = roc_auc_score(y_true, prob)
        fpr, tpr, _ = roc_curve(y_true, prob)
        mn = normalize_model_name(model_name)

        ax.plot(
            fpr, tpr,
            linewidth=2.0,
            color=get_model_color(mn),
            label=f"{mn} = {auc:.3f}"
        )

    ax.set_title(panel_label, fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.20)

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=12,
        handlelength=1.8,
        borderpad=0.4,
        labelspacing=0.3
    )
    return ax

def plot_roc_panels_with_cm(
    y_true,
    oof_prob_by_featureset_by_model,
    roc_panel_order,
    cm,
    cm_title="Confusion Matrix",
    cm_labels=("Healthy", "Depression"),
    ncols=2,
    figsize=(10.0, 7.6),
):
    items = [("roc", fs) for fs in roc_panel_order] + [("cm", None)]

    n_panels = len(items)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(-1)

    for i, (kind, value) in enumerate(items):
        ax = axes[i]

        if kind == "roc":
            fs_name = value
            by_model = oof_prob_by_featureset_by_model.get(fs_name, {})
            plot_roc_all_models_for_featureset(
                y_true=y_true,
                oof_prob_by_model=by_model,
                panel_label=display_featureset_name(fs_name),
                ax=ax,
            )

        else:
            plot_confusion_matrix_heatmap(
                cm=cm,
                title=cm_title,
                labels=cm_labels,
                normalize=False,
                show_f1=True,
                positive_index=1,
                ax=ax,
            )

    for j in range(len(items), len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.96,
        wspace=0.18,
        hspace=0.24,
    )
    return fig

def plot_roc_panels_by_featureset(
    y_true,
    oof_prob_by_featureset_by_model,
    panel_order,
    ncols=3,
    figsize=(15.0, 4.8)
):
    n_panels = len(panel_order)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(-1)

    last_i = -1
    for i, fs_name in enumerate(panel_order):
        last_i = i
        by_model = oof_prob_by_featureset_by_model.get(fs_name, {})
        plot_roc_all_models_for_featureset(
            y_true=y_true,
            oof_prob_by_model=by_model,
            panel_label=display_featureset_name(fs_name),
            ax=axes[i],
        )

    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.12,
        top=0.90,
        wspace=0.24,
        hspace=0.32,
    )
    return fig

def plot_feature_sets_for_model(df_summary, model_name, feature_sets, title,
                                save_path=None, dpi=DPI, rotate_xticks=20, wrap_width=18,
                                demographic_baseline=None, posthoc_feature_sets=None):
    df = df_summary.copy()
    df["Model_norm"] = df["Model"].map(normalize_model_name)
    model_name = normalize_model_name(model_name)

    ordered_feature_sets = list(feature_sets)
    if demographic_baseline and np.isfinite(demographic_baseline.get("auroc", np.nan)):
        if DEMOGRAPHIC_BASELINE_KEY not in ordered_feature_sets:
            ordered_feature_sets = [DEMOGRAPHIC_BASELINE_KEY] + ordered_feature_sets

    sub = df[
        (df["Model_norm"] == model_name) &
        (df["FeatureSet"].isin([fs for fs in ordered_feature_sets if fs != DEMOGRAPHIC_BASELINE_KEY]))
    ].copy()

    if demographic_baseline and np.isfinite(demographic_baseline.get("auroc", np.nan)):
        demo_row = pd.DataFrame([{
            "FeatureSet": DEMOGRAPHIC_BASELINE_KEY,
            "Model": "Baseline",
            "Model_norm": "Baseline",
            "AUROC": float(demographic_baseline["auroc"]),
            "AUROC_lo": float(demographic_baseline["auroc_lo"]),
            "AUROC_hi": float(demographic_baseline["auroc_hi"]),
            "n_features": len(demographic_baseline.get("features", [])),
        }])
        sub = pd.concat([demo_row, sub], ignore_index=True)

    if len(sub) == 0:
        print(f"[PLOT] No rows for model={model_name}")
        return

    sub["FeatureSet"] = pd.Categorical(sub["FeatureSet"], categories=ordered_feature_sets, ordered=True)
    sub = sub.sort_values("FeatureSet").reset_index(drop=True)

    x = np.arange(len(sub))
    y = sub["AUROC"].to_numpy(dtype=float)
    lo = sub["AUROC_lo"].to_numpy(dtype=float)
    hi = sub["AUROC_hi"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    # 세로 길이는 약간 줄임
    fig, ax = plt.subplots(figsize=(max(7.2, 1.10 * len(sub)), 5.4))
    bars = ax.bar(x, y, width=0.7, yerr=yerr, capsize=6, edgecolor="black", linewidth=1.0)

    fs_ordered = sub["FeatureSet"].astype(str).tolist()
    for bar, fs in zip(bars, fs_ordered):
        style = get_featureset_bar_style(fs)
        bar.set_facecolor(style["facecolor"])
        bar.set_edgecolor(style["edgecolor"])
        bar.set_linewidth(style["linewidth"])
        bar.set_linestyle(style["linestyle"])
        bar.set_alpha(style["alpha"])
        if style["hatch"] is not None:
            bar.set_hatch(style["hatch"])

    # 핵심: AUROC 구간만 확대해서 보여주기
    y_min, y_max = apply_zoomed_auroc_axis(
        ax, lo, hi,
        tick_step=0.05,   # tick 간격 조금 넓게
        min_floor=0.50,
        top_pad=0.03
    )

    if y_min <= 0.5 <= y_max:
        ax.axhline(0.5, linestyle="--", linewidth=1.5, alpha=0.75, color="#666666")

    nfeat_col = infer_nfeat_col(sub)
    value_text_y_pad = 0.015 * (y_max - y_min)
    bottom_text_y = y_min + 0.02 * (y_max - y_min)

    for i, row in enumerate(sub.itertuples(index=False)):
        fs = fs_ordered[i]
        style = get_featureset_bar_style(fs)

        ax.text(
            i,
            min(hi[i] + value_text_y_pad, y_max - 0.01 * (y_max - y_min)),
            f"{y[i]:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

        if nfeat_col is not None:
            n_feat_val = getattr(row, nfeat_col, None)
            n_min = getattr(row, "n_features_min", None)
            n_max = getattr(row, "n_features_max", None)
            if n_feat_val is not None and pd.notna(n_feat_val):
                if (n_min is not None and n_max is not None and pd.notna(n_min) and pd.notna(n_max) and int(n_min) != int(n_max)):
                    feat_label = f"var={int(n_min)}-{int(n_max)}"
                else:
                    feat_label = f"var={int(n_feat_val)}"
                ax.text(
                    i, bottom_text_y, feat_label,
                    ha="center", va="bottom", fontsize=14,
                    color=style["text_color"], fontweight="bold",)

    def wrap_label(text, width=18):
        words = text.replace("_", " ").split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + (1 if current else 0) <= width:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return "\n".join(lines)

    labels = [wrap_label(display_featureset_name(fs), wrap_width) for fs in fs_ordered]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # --- post hoc 구분 표시 추가 ---
    posthoc_feature_sets = set(map(str, posthoc_feature_sets or []))
    posthoc_mask = sub["FeatureSet"].astype(str).isin(posthoc_feature_sets)
    if posthoc_mask.any():
        first_posthoc_idx = np.where(posthoc_mask)[0][0]
        if first_posthoc_idx > 0:
            ax.axvline(first_posthoc_idx - 0.5, color="gray",
                       linestyle=(0, (3, 3)), linewidth=1.2, alpha=0.9, zorder=0)
        # ax.axvspan(first_posthoc_idx - 0.5, len(sub) - 0.5,
        #            color="gray", alpha=0.06, zorder=0)
        ax.text((first_posthoc_idx + len(sub) - 1) / 2,
                y_max + 0.005 * (y_max - y_min),
                "Post hoc",
                ha="center", va="bottom", fontsize=11,
                color="#555555", fontweight="bold", clip_on=False)
    
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def plot_bootstrap_hist(samples, title, save_path=None, dpi=DPI):
    samples = np.asarray(samples, dtype=float).ravel()
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        print("[BOOTSTRAP] No valid samples to plot.")
        return

    lo = np.percentile(samples, 2.5)
    hi = np.percentile(samples, 97.5)
    mu = samples.mean()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(samples, bins=30, alpha=0.85)
    ax.axvline(mu, linewidth=2, label=f"Mean={mu:.3f}")
    ax.axvline(lo, linestyle="--", linewidth=2, label=f"2.5%={lo:.3f}")
    ax.axvline(hi, linestyle="--", linewidth=2, label=f"97.5%={hi:.3f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Bootstrap metric value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def collect_probs_for_panels(store, feature_sets):
    y_true = None
    probs = {}
    for (fs_name, model_name), res in store.items():
        if fs_name not in feature_sets or "oof_prob" not in res or "y" not in res:
            continue
        y = np.asarray(res["y"]).astype(int).ravel()
        try:
            prob = as_binary_score_1d(res["oof_prob"])
        except Exception:
            continue
        if y_true is None:
            y_true = y
        elif len(y) != len(y_true) or not np.array_equal(y, y_true):
            continue
        probs.setdefault(fs_name, {})[normalize_model_name(model_name)] = prob
    return y_true, probs

def plot_cumulative_curve(df_cum, title=None, save_path=None, dpi=DPI, plateau_delta=0.02):
    if df_cum is None or len(df_cum) == 0:
        print("[PLOT] df_cum is empty. Skip.")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    days = df_cum["days"].to_numpy(dtype=float)
    auroc = df_cum["auroc"].to_numpy(dtype=float)
    lo = df_cum["auroc_lo"].to_numpy(dtype=float)
    hi = df_cum["auroc_hi"].to_numpy(dtype=float)

    main_color = "#3587FC"
    ref_color = "0.55"   # 회색 기준선
    text_color = "0.15"

    ax.fill_between(days, lo, hi, color=main_color, alpha=0.16, linewidth=0)
    ax.plot(
        days, auroc,
        "o-",
        color=main_color,
        linewidth=2.4,
        markersize=6.8,
        markerfacecolor=main_color,
        markeredgecolor=main_color,
    )

    ax.axhline(0.5, linestyle="--", linewidth=1.2, color=ref_color, alpha=0.9)

    final_auc = float(auroc[-1])
    plateau_idx = np.where(auroc >= final_auc - float(plateau_delta))[0]

    if len(plateau_idx) > 0:
        idx0 = int(plateau_idx[0])
        plateau_day = days[idx0]
        plateau_auc = auroc[idx0]
        if plateau_day >= float(days[-1]) - 1:
            text_dx = -0.35
            text_ha = "right"
        elif plateau_day >= float(days[-1]) - 2:
            text_dx = -0.45
            text_ha = "left"
        else:
            text_dx = 0.9
            text_ha = "left"

        ax.axvline(
            plateau_day,
            linestyle=":",
            linewidth=1.5,
            color=main_color,
            alpha=0.75
        )

        ax.annotate(
            f"Plateau ≈ day {int(plateau_day)}",
            xy=(plateau_day, plateau_auc),
            xytext=(plateau_day + text_dx, plateau_auc - 0.055),
            fontsize=13,
            color=text_color,
            ha=text_ha,
            arrowprops=dict(
                arrowstyle="->",
                lw=1.2,
                color=text_color
            ),
        )

    ax.set_xlabel("Cumulative days", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=13.5)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(0.5, float(days[-1]) + 0.5)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(days)
    ax.grid(alpha=0.22)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return ax

def plot_cumulative_curve_stacked(df_cum, title, save_path=None, dpi=DPI, plateau_delta=0.02):
    if df_cum is None or len(df_cum) == 0:
        print("[PLOT] df_cum is empty. Skip.")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    days = df_cum["days"].to_numpy(dtype=float)
    auroc = df_cum["auroc"].to_numpy(dtype=float)
    lo = df_cum["auroc_lo"].to_numpy(dtype=float)
    hi = df_cum["auroc_hi"].to_numpy(dtype=float)

    ax1.fill_between(days, lo, hi, alpha=0.20)
    ax1.plot(days, auroc, "o-", linewidth=2.2, markersize=6, label="AUROC")
    ax1.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7)

    final_auc = float(auroc[-1])
    plateau_idx = np.where(auroc >= final_auc - float(plateau_delta))[0]
    if len(plateau_idx) > 0:
        plateau_day = days[int(plateau_idx[0])]
        ax1.axvline(plateau_day, linestyle=":", linewidth=1.5, alpha=0.8)

    ax1.set_ylabel("AUROC", fontsize=13)
    ax1.set_title("Cumulative AUROC (H vs D)", fontweight="bold", fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(loc="lower right", fontsize=11)

    p_h = df_cum["p_h"].to_numpy(dtype=float)
    p_d = df_cum["p_d"].to_numpy(dtype=float)
    ax2.plot(days, p_h, "s-", linewidth=2, markersize=5, label="Healthy (OOF)")
    ax2.plot(days, p_d, "o-", linewidth=2, markersize=5, label="Depression (OOF)")

    ordered_mask = df_cum["ordered"].fillna(False).to_numpy(dtype=bool)
    for i in range(len(days)):
        if ordered_mask[i]:
            ax2.axvspan(days[i] - 0.4, days[i] + 0.4, alpha=0.06)

    ax2.set_xlabel("Cumulative Days", fontsize=14)
    ax2.set_ylabel("Mean P(depression)", fontsize=13)
    ax2.set_title("Group mean probabilities over time", fontweight="bold", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.legend(loc="upper left", fontsize=11)

    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0.5, float(days[-1]) + 0.5)
    ax2.set_xticks(days)

#    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig

def plot_sliding_window(df_slide, title, save_path=None, dpi=DPI):
    if df_slide is None or len(df_slide) == 0:
        print("[PLOT] df_slide empty. Skip.")
        return None

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    markers = ["o", "s", "^", "D", "v"]

    window_sizes = sorted(df_slide["window_size"].unique())
    palette = sns.color_palette("Set2", n_colors=len(window_sizes))  # 여기

    for i, window_size in enumerate(window_sizes):
        sub = df_slide[df_slide["window_size"] == window_size].sort_values("start_day")
        midpoint = sub["start_day"].to_numpy(dtype=float) + (window_size - 1) / 2.0
        color = palette[i]

        ax.fill_between(
            midpoint,
            sub["auroc_lo"].values,
            sub["auroc_hi"].values,
            color=color,
            alpha=0.12
        )
        ax.plot(
            midpoint,
            sub["auroc"].values,
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=3,
            markersize=5,
            color=color,
            label=f"{window_size}-day window"
        )

    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7, color="gray")
    ax.set_xlabel("Window Midpoint (Day)", fontsize=13)
    ax.set_ylabel("AUROC", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.35, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig


def plot_selection_heatmap(matrix, title="Feature Selection Stability Heatmap", save_path=None, dpi=300):
    if matrix.empty:
        return

    arr = matrix.values.astype(int)
    fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.3), min(12, len(matrix) * 0.15 + 2)))
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Fold", fontsize=10)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.5, label="Selected (1) / Not (0)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def plot_reliability_diagram(y_true, y_prob, n_bins=10, title="Reliability Diagram", save_path=None, dpi=300):
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    except ValueError:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=min(5, n_bins), strategy="uniform")

    brier = float(brier_score_loss(y_true, y_prob))
    ece = ece_quantile(y_true, y_prob, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, "o-", linewidth=2, markersize=6, label="Model")
    ax1.set_xlabel("Mean predicted probability")  
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Reliability Diagram")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Healthy", color="#2ca02c")
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Depression", color="#d62728")
    ax2.set_xlabel("Predicted P(Depression)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Probability Distribution\nBrier={brier:.3f} | ECE={ece:.3f}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

import re

def clean_legend_label(label):
    return re.sub(r"\s*\(var=.*?\)", "", label).strip()


def plot_cumulative_dual(
    df_cum_stable,
    df_cum_combined,
    label_stable,
    label_combined,
    title,
    save_path=None,
    dpi=DPI,
    plateau_delta=0.02,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    configs = [
        (
            df_cum_stable,
            label_stable,
            COMPARISON_COLORS["Combined_stable"],
            "--", 2.2,
            "D", 4.5, 0.95, 2,
        ),
        (
            df_cum_combined,
            label_combined,
            COMPARISON_COLORS["Combined_main"],
            "-", 2.8,
            "o", 6, 1.0, 3,
        ),
    ]

    for (df_c, label, line_color, ls, lw, marker, ms, alpha, zo) in configs:
        if df_c is None or len(df_c) == 0:
            continue

        days = df_c["days"].to_numpy(dtype=float)
        auroc = df_c["auroc"].to_numpy(dtype=float)
        lo = df_c["auroc_lo"].to_numpy(dtype=float)
        hi = df_c["auroc_hi"].to_numpy(dtype=float)

        ax.fill_between(
            days, lo, hi,
            alpha=0.10 * alpha,
            color=line_color,
            zorder=zo - 1
        )

        ax.plot(
            days, auroc,
            linestyle=ls,
            linewidth=lw,
            marker=marker,
            markersize=ms,
            color=line_color,
            label=clean_legend_label(label),
            alpha=alpha,
            zorder=zo,
        )

    if df_cum_combined is not None and len(df_cum_combined) > 0:
        days_c = df_cum_combined["days"].to_numpy(dtype=float)
        auroc_c = df_cum_combined["auroc"].to_numpy(dtype=float)

        milestones = {
            2: "Day 2",
            4: "Day 4",
            7: "Day 7",
            10: "Day 10",
            14: "Day 14",
        }

        offsets = {
            2: (1.0, -0.06),
            4: (1.0, -0.085),
            7: (1.0, -0.06),
            10: (1.0, -0.06),
            14: (0.25, -0.06),
        }

        for d_ms, lbl in milestones.items():
            idx_ms = np.where(days_c == d_ms)[0]
            if len(idx_ms) > 0:
                i = idx_ms[0]
                dx, dy = offsets.get(d_ms, (1.0, -0.06))

                ax.annotate(
                    f"{lbl}\n{auroc_c[i]:.3f}",
                    xy=(d_ms, auroc_c[i]),
                    xytext=(d_ms + dx, auroc_c[i] + dy),
                    fontsize=13,
                    fontweight="bold",
                    color=COMPARISON_COLORS["Combined_main"],
                    arrowprops=dict(
                        arrowstyle="->",
                        color=COMPARISON_COLORS["Combined_main"],
                        lw=1.2,
                    ),
                    zorder=10,
                )

    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.5, color="#999999")
    ax.set_xlabel("Cumulative Days", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=13.5)
    ax.tick_params(axis="both", labelsize=13)
    #ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.35, 1.0)
    ax.grid(alpha=0.2)

  #  ax.legend(fontsize=12, loc="lower right", framealpha=0.9, edgecolor="gray")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig
    

def plot_sliding_dual(df_slide_stable, df_slide_combined, label_stable, label_combined,
                      title, save_path=None, dpi=300):
    all_ws = sorted(set(
        list(df_slide_stable["window_size"].unique() if df_slide_stable is not None and len(df_slide_stable) else []) +
        list(df_slide_combined["window_size"].unique() if df_slide_combined is not None and len(df_slide_combined) else [])
    ))
    if not all_ws:
        return None

    fig, axes = plt.subplots(1, len(all_ws), figsize=(5.5 * len(all_ws), 4.5), sharey=True)
    if len(all_ws) == 1:
        axes = [axes]

    for ax, ws in zip(axes, all_ws):
        for df_s, label, color, fill_color, marker in [
            (df_slide_stable, label_stable,
             COMPARISON_COLORS["Combined_stable"], COMPARISON_FILL_COLORS["Combined_stable"], "o"),
            (df_slide_combined, label_combined,
             COMPARISON_COLORS["Combined_main"], COMPARISON_FILL_COLORS["Combined_main"], "s"),
        ]:
            if df_s is None or len(df_s) == 0:
                continue
            sub = df_s[df_s["window_size"] == ws].sort_values("start_day")
            if len(sub) == 0:
                continue
            midpoint = sub["start_day"].to_numpy(dtype=float) + (ws - 1) / 2.0
            ax.fill_between(midpoint, sub["auroc_lo"].values, sub["auroc_hi"].values,
                            alpha=0.14, color=fill_color)
            ax.plot(midpoint, sub["auroc"].values, f"{marker}-", linewidth=2, markersize=4,
                    color=color, label=label)

        ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7, color="#666666")
        ax.set_title(f"{ws}-day window", fontweight="bold")
        ax.set_xlabel("Window Midpoint (Day)")
        ax.set_xlim(0.5, 15.5)
        ax.set_ylim(0.35, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("AUROC")
 #   fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig
    


# =============================================================================
# Main pipeline — Stable features 자동 추출 통합
# =============================================================================
def run_main_pipeline(paths, out_dir, cfg, model_whitelist=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    # 데이터 로드
    df_main, acoustic_cols, lexical_cols = load_and_build_day_level(paths, cfg)

    # 모델 빌드
    clf_models, reg_models = build_models(cfg.seed)
    whitelist = normalize_model_whitelist(model_whitelist)
    if whitelist:
        clf_models = {k: v for k, v in clf_models.items() if k in whitelist}
        reg_models = {k: v for k, v in reg_models.items() if k in whitelist}
    common_models = [m for m in MODEL_ORDER if m in clf_models and m in reg_models]
    print(f"\n[MODELS] {common_models}")

    # Step 1: 3개 기본 feature set으로 평가 (Stable 추출 전)
    feature_sets_base = build_feature_sets(acoustic_cols, lexical_cols, cfg, stable_features=None)

    # Step 2: Combined_main에서 Stable features 자동 추출
    df_prt_full = aggregate_participants(df_main, acoustic_cols, lexical_cols, cfg)
    stable_features, jaccard, sel_matrix = extract_stable_features(
        df_prt_full, feature_sets_base["Combined_main"], clf_models["ElasticNet"], cfg,
    )

    # [추가] demographic baseline 미리 계산
    demo_baseline = evaluate_demographic_baseline(
        df_prt_full,
        outer_splits=cfg.outer_splits,
        outer_repeats=cfg.outer_repeats,
        seed=cfg.seed,
        n_bootstrap=cfg.n_bootstrap,
    )

    # Step 3: Stable features를 포함한 4개 feature set으로 최종 평가
    feature_sets = build_feature_sets(acoustic_cols, lexical_cols, cfg, stable_features=stable_features)
    print("\n[FEATURE SETS]")
    for name, feats in feature_sets.items():
        print(f"  {display_featureset_name(name)}: {len(feats)} features")

    # Step 4: 모든 조합 평가
    rows = []
    report_rows = []
    store = {}
    print_header("RUN EVALUATION (H vs D)", width=70)
    print(f"Outer: {cfg.outer_repeats}×{cfg.outer_splits} | Inner(thr): {cfg.inner_splits_thr}")

    for fs_name, raw_features in feature_sets.items():
        df_prt = aggregate_participants(df_main, acoustic_cols, lexical_cols, cfg)
        features = filter_existing_features(df_prt, raw_features)
        stable_eval_mode = (fs_name == "Combined_stable" and cfg.nested_stable_eval)

        if stable_eval_mode:
            candidate_features = filter_existing_features(df_prt, feature_sets["Combined_main"])
            if len(candidate_features) < cfg.stable_min_features:
                print(f"  [SKIP] {display_featureset_name(fs_name)}: too few candidate features ({len(candidate_features)})")
                continue
            print(
                f"\n  {display_featureset_name(fs_name)} | "
                f"nested re-selection from {len(candidate_features)} candidate features"
            )
        else:
            if len(features) < 3:
                print(f"  [SKIP] {display_featureset_name(fs_name)}: too few features ({len(features)})")
                continue
            print(f"\n  {display_featureset_name(fs_name)} | n_feats={len(features)}")

        for model_name in common_models:
            print(f"    {model_name} ...", end="", flush=True)
            try:
                if stable_eval_mode:
                    res = evaluate_hd_oof_nested_stable(
                        df_prt,
                        feature_sets["Combined_main"],
                        clf_models[model_name],
                        reg_models[model_name],
                        clf_models["ElasticNet"],
                        cfg,
                        model_name=model_name,
                    )
                else:
                    res = evaluate_hd_oof(
                        df_prt,
                        features,
                        clf_models[model_name],
                        reg_models[model_name],
                        cfg,
                        model_name=model_name,
                    )
            except Exception as exc:
                print(f" FAILED ({exc})")
                continue

            metrics = res["metrics"]
            thr = res["thresholds"]
            report = res["classification_report"]
            n_features_report = int(res.get("n_features_report", len(features)))
            n_feat_min = int(res.get("n_features_min", n_features_report))
            n_feat_max = int(res.get("n_features_max", n_features_report))

            rows.append({ "FeatureSet": fs_name,
                         "Model": model_name,
                         "n_features": n_features_report,
                         "n_features_min": n_feat_min,
                         "n_features_max": n_feat_max,
                         **metrics,
                         "Thr_global": thr["thr_global"],
                         "Thr_mean": thr["Thr_mean"] if "Thr_mean" in thr else thr["thr_mean"],
                         "Thr_std": thr["Thr_std"] if "Thr_std" in thr else thr["thr_std"],  })

            report_rows.append({"FeatureSet": fs_name, "Model": model_name, "n_features": n_features_report, **report, })
            store[(fs_name, model_name)] = res
            print(f" AUROC={metrics['AUROC']:.3f} [{metrics['AUROC_lo']:.3f}-{metrics['AUROC_hi']:.3f}]"
                  f" | Sens={report['Sensitivity']:.3f}"
                  f" | Spec={report['Specificity']:.3f}"
                  f" | F1={report['F1']:.3f}")

    df_summary = pd.DataFrame(rows).sort_values(["FeatureSet", "AUROC"], ascending=[True, False])
    df_summary.to_csv(out_dir / f"summary_{ts}.csv", index=False)
    df_report = pd.DataFrame(report_rows)
    if len(df_report) > 0:
        df_report = df_report.sort_values(["FeatureSet", "AUROC"], ascending=[True, False])
        df_report.to_csv(out_dir / f"classification_reports_{ts}.csv", index=False)

    # Best prespecified model info
    best_info = {}
    if len(df_summary) > 0:
        df_best_pick = df_summary[df_summary["FeatureSet"].isin(["Acoustic_mean+std", "Lexical_mean", "Combined_main"])].copy()
        if len(df_best_pick) == 0:
            df_best_pick = df_summary.copy()
        df_best_pick["Model_norm"] = df_best_pick["Model"].map(normalize_model_name)
        model_rank_map = {m: i for i, m in enumerate(MODEL_ORDER)}
        df_best_pick["model_rank"] = df_best_pick["Model_norm"].map(lambda x: model_rank_map.get(x, 999))
        best = (
            df_best_pick
            .sort_values(["AUROC", "model_rank"], ascending=[False, True])
            .iloc[0]
        )
        best_info = {
            "best_fs": str(best["FeatureSet"]),
            "best_model": str(best["Model"]),
            "best_auc": float(best["AUROC"]),
            "best_ts": ts,
        }
        print_header("BEST PRESPECIFIED MODEL", width=70)
        print(f"  {display_featureset_name(best_info['best_fs'])} + {best_info['best_model']}")
        print(f"  AUROC: {best['AUROC']:.3f} [{best['AUROC_lo']:.3f}-{best['AUROC_hi']:.3f}]")

        if cfg.make_supp_plots:
            available = df_summary["FeatureSet"].unique().tolist()
            fs_order_main = [fs for fs in MAIN_FEATURESET_PLOT_ORDER if fs in available or fs == DEMOGRAPHIC_BASELINE_KEY]
            fs_order_supp = [fs for fs in SUPPLEMENTARY_FEATURESET_PLOT_ORDER if fs in available or fs == DEMOGRAPHIC_BASELINE_KEY]
            feature_compare_model = "ElasticNet" if "ElasticNet" in common_models else best_info["best_model"]

            plot_feature_sets_for_model(
                df_summary=df_summary,
                model_name=feature_compare_model,
                feature_sets=fs_order_main,
                title=f"Prespecified Feature Set Comparison ({feature_compare_model})",
                save_path=out_dir / f"FeatureSet_Compare_{feature_compare_model}_{ts}.png",
                dpi=cfg.dpi,
                demographic_baseline=demo_baseline,
                posthoc_feature_sets=[],
            )

            y_true, probs = collect_probs_for_panels(store,
                                                     [fs for fs in fs_order_supp if fs != DEMOGRAPHIC_BASELINE_KEY])
            if y_true is not None and len(probs) >= 2:
                roc_order = [fs for fs in fs_order_supp if fs in probs and fs != "Combined_stable"]
                best_key = (best_info["best_fs"], best_info["best_model"])
                res_best = store.get(best_key)
                
                fig = plot_roc_panels_by_featureset(
                    y_true=y_true,
                    oof_prob_by_featureset_by_model=probs,
                    panel_order=roc_order if len(roc_order) > 0 else [fs for fs in fs_order_supp if fs in probs],
                    ncols=3,
                    figsize=(15.0, 4.8),
                )
                fig.savefig(out_dir / f"ROC_Panels_AllModels_{ts}.png", dpi=cfg.dpi, bbox_inches="tight")
                plt.show()

                if res_best is not None:
                    plot_confusion_matrix_heatmap(
                        cm=res_best["confusion_matrix"],
                        title=None,
                        labels=("Healthy", "Depression"),
                        normalize=False,
                        save_path=out_dir / f"ConfusionMatrix_Best_{best_info['best_fs']}_{best_info['best_model']}_{ts}.png",
                        dpi=cfg.dpi,
                        show_f1=True,
                        positive_index=1,
                    )

        if cfg.make_bootstrap_plots:
            key_best = (best_info["best_fs"], best_info["best_model"])
            res_best = store.get(key_best)
            if res_best:
                auc_samples = cluster_bootstrap_samples(
                    metric_func=lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
                    y_true=res_best["y"], y_pred=res_best["oof_prob"], groups=res_best["prt_ids"],
                    n_bootstrap=max(cfg.n_bootstrap_plot, cfg.n_bootstrap),
                    seed=cfg.seed + 999,
                )
                plot_bootstrap_hist(
                    auc_samples,
                    title=f"Bootstrap AUROC\n{display_featureset_name(best_info['best_fs'])} | {best_info['best_model']}",
                    save_path=out_dir / f"Bootstrap_AUROC_{best_info['best_fs']}_{best_info['best_model']}_{ts}.png",
                    dpi=cfg.dpi,
                )

        best_key = (best_info["best_fs"], best_info["best_model"])
        res_best = store.get(best_key)
        if res_best is not None:
            cm_df = pd.DataFrame(  res_best["confusion_matrix"],
                                 index=["True_Healthy", "True_Depression"],
                                 columns=["Pred_Healthy", "Pred_Depression"],)
            cm_df.to_csv(out_dir / f"ConfusionMatrix_Best_{best_info['best_fs']}_{best_info['best_model']}_{ts}.csv")

            pd.DataFrame([res_best["classification_report"]]).to_csv(
                out_dir / f"ClassificationReport_Best_{best_info['best_fs']}_{best_info['best_model']}_{ts}.csv", index=False,)

    print(f"\n  [Stability] Exploratory global Jaccard = {jaccard:.3f}")
    print(
        f"  [Reduced model] Exploratory globally selected features "
        f"({len(stable_features)}): {stable_features}"
    )

    # Combined_stable의 nested fold별 feature 수 range 수집
    stable_nfeat_min = len(stable_features)
    stable_nfeat_max = len(stable_features)
    for (fs_name_k, _), res_k in store.items():
        if fs_name_k == "Combined_stable" and "n_features_min" in res_k:
            stable_nfeat_min = min(stable_nfeat_min, int(res_k["n_features_min"]))
            stable_nfeat_max = max(stable_nfeat_max, int(res_k["n_features_max"]))

    pack = {
        "df_main": df_main,
        "acoustic_cols": acoustic_cols,
        "lexical_cols": lexical_cols,
        "clf_models": clf_models,
        "reg_models": reg_models,
        "best_info": best_info,
        "stable_features": stable_features,
        "combined_main_features": feature_sets["Combined_main"],
        "jaccard": jaccard,
        "sel_matrix": sel_matrix,
        "demo_baseline": demo_baseline,
        "nested_stable_eval": cfg.nested_stable_eval,
        "stable_nfeat_min": stable_nfeat_min,
        "stable_nfeat_max": stable_nfeat_max,
    }

    return df_summary, store, pack

# =============================================================================
# Temporal analysis — Stable features + Combined_main 두 가지 모두 수행
# =============================================================================
def _run_temporal_one_featureset(df_main, acoustic_cols, lexical_cols, cfg, clf,
                                 features, fs_label, out_dir, max_day, min_days_per_prt,
                                 window_sizes, early_split_day,
                                 selector_clf=None, candidate_features=None, model_name="ElasticNet"):
    """단일 feature set에 대한 temporal 분석 (cumulative + sliding)."""
    ts = now_ts()

    print_header(f"TEMPORAL [{fs_label}]: Cumulative Days-to-Detection")
    cum_results = []
    for k in range(1, max_day + 1):
        df_prt = aggregate_participants(df_main, acoustic_cols, lexical_cols, cfg,
                                       day_range=(1, k), min_days_per_prt=min_days_per_prt)
        if len(df_prt) < 20:
            continue
        if selector_clf is not None and candidate_features is not None:
            res = evaluate_temporal_point_nested_stable(df_prt, candidate_features, selector_clf, clf, cfg, model_name=model_name)
        else:
            res = evaluate_temporal_point(df_prt, features, clf, cfg)
        res["days"] = k
        res["day_range"] = f"1-{k}"
        cum_results.append(res)
        print(
            f"  Days 1-{k:2d}: AUROC={res['auroc']:.3f} [{res['auroc_lo']:.3f}-{res['auroc_hi']:.3f}]"
            f" | H={res['p_h']:.3f} D={res['p_d']:.3f} | Ordered={res['ordered']}"
        )
    df_cum = pd.DataFrame(cum_results)
    df_cum.to_csv(out_dir / f"temporal_cumulative_{fs_label}_{ts}.csv", index=False)

    print_header(f"TEMPORAL [{fs_label}]: Sliding Window Analysis")
    slide_results = []
    for ws in window_sizes:
        for start in range(1, max_day - ws + 2):
            end = start + ws - 1
            df_prt = aggregate_participants(df_main, acoustic_cols, lexical_cols, cfg,
                                           day_range=(start, end), min_days_per_prt=min_days_per_prt)
            if len(df_prt) < 20:
                continue
            if selector_clf is not None and candidate_features is not None:
                res = evaluate_temporal_point_nested_stable(df_prt, candidate_features, selector_clf, clf, cfg, model_name=model_name)
            else:
                res = evaluate_temporal_point(df_prt, features, clf, cfg)
            res.update({"window_size": ws, "start_day": start, "end_day": end, "day_range": f"{start}-{end}"})
            slide_results.append(res)
            print(f"  Days {start:2d}-{end:2d} (w={ws}): AUROC={res['auroc']:.3f}")
    df_slide = pd.DataFrame(slide_results)
    df_slide.to_csv(out_dir / f"temporal_sliding_{fs_label}_{ts}.csv", index=False)

    return {"cumulative": df_cum, "sliding": df_slide}

def run_temporal_analysis(df_main, acoustic_cols, lexical_cols, cfg, clf_models,
                          stable_features, out_dir, max_day=15, min_days_per_prt=1,
                          window_sizes=None, early_split_day=8,
                          stable_nfeat_min=None, stable_nfeat_max=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()
    window_sizes = window_sizes or [3, 5, 7]
    clf = clf_models["ElasticNet"]

    feature_sets = build_feature_sets(acoustic_cols, lexical_cols, cfg, stable_features=stable_features)
    combined_main_features = feature_sets["Combined_main"]
    stable_nested_mode = bool(cfg.nested_stable_eval)

    # legend label에 feature 수 range 포함 (var= 표기)
    if (stable_nfeat_min is not None and stable_nfeat_max is not None
            and stable_nfeat_min != stable_nfeat_max):
        stable_label = f"{display_featureset_name('Combined_stable')} (var={stable_nfeat_min}-{stable_nfeat_max})"
    else:
        stable_label = f"{display_featureset_name('Combined_stable')} (var={len(stable_features)})"
    combined_label = f"{display_featureset_name('Combined_main')} (var={len(combined_main_features)})"

    print_header("TEMPORAL ANALYSIS", width=70)
    print("  Model: ElasticNet (fixed)")
    if stable_nested_mode:
        print(f"  Feature set 1: {stable_label} | nested re-selection from {len(combined_main_features)} candidates")
    else:
        print(f"  Feature set 1: {stable_label} ({len(stable_features)} features)")
    print(f"  Feature set 2: {combined_label} ({len(combined_main_features)} features)")

    # A) Combined_stable temporal
    res_stable = _run_temporal_one_featureset(
        df_main, acoustic_cols, lexical_cols, cfg, clf,
        stable_features, "Combined_stable", out_dir, max_day,
        min_days_per_prt, window_sizes, early_split_day,
        selector_clf=clf_models["ElasticNet"] if stable_nested_mode else None,
        candidate_features=combined_main_features if stable_nested_mode else None,
        model_name="ElasticNet",
    )

    plot_sliding_window(
        res_stable["sliding"],
        title=f"Sliding Windows | {stable_label} | ElasticNet",
        save_path=out_dir / f"Temporal_Sliding_Reduced_{ts}.png", dpi=cfg.dpi,
    )

    # B) Combined_main temporal
    res_combined = _run_temporal_one_featureset(
        df_main, acoustic_cols, lexical_cols, cfg, clf,
        combined_main_features, "Combined_main", out_dir, max_day,
        min_days_per_prt, window_sizes, early_split_day,
    )

    plot_cumulative_curve(
        res_combined["cumulative"],
        title=f"Cumulative AUROC | {combined_label} | ElasticNet",
        save_path=out_dir / f"Temporal_Cumulative_Combined_{ts}.png", dpi=cfg.dpi,
    )
    plot_sliding_window(
        res_combined["sliding"],
        title=f"Sliding Windows | {combined_label} | ElasticNet",
        save_path=out_dir / f"Temporal_Sliding_Combined_{ts}.png", dpi=cfg.dpi,
    )

    # C) Dual comparison plots
    label_s = stable_label
    label_c = combined_label

    plot_cumulative_dual(
        res_stable["cumulative"], res_combined["cumulative"],
        label_stable=label_s, label_combined=label_c,
        title="Cumulative AUROC Comparison | ElasticNet",
        save_path=out_dir / f"Temporal_Cumulative_DUAL_{ts}.png", dpi=cfg.dpi,
    )

    return {"stable": res_stable, "combined_main": res_combined}


# =============================================================================
# Demographic baseline
# =============================================================================
def evaluate_demographic_baseline(df_prt, outer_splits=3, outer_repeats=20, seed=42, n_bootstrap=1000):
    df_hd = df_prt[df_prt["group"].isin(["healthy", "depression"])].copy()
    if len(df_hd) == 0:
        return {}

    y = (df_hd["group"].values == "depression").astype(int)
    prt_ids = df_hd["prt"].values
    demo_feats = [c for c in ["age", "gender"] if c in df_hd.columns]
    if not demo_feats:
        return {}

    X = df_hd[demo_feats].to_numpy(dtype=float)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)),
    ])

    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    oof_sum = np.zeros(len(y), dtype=float)
    oof_cnt = np.zeros(len(y), dtype=float)
    for tr, te in rskf.split(X, y):
        m = clone(pipe)
        m.fit(X[tr], y[tr])
        oof_sum[te] += m.predict_proba(X[te])[:, 1]
        oof_cnt[te] += 1.0
    oof = oof_sum / np.clip(oof_cnt, 1, None)

    auc = safe_auc(y, oof)
    _, lo, hi = cluster_bootstrap_ci(
        lambda a, b: roc_auc_score(a, b) if len(np.unique(a)) > 1 else float("nan"),
        y, oof, prt_ids, n_bootstrap=n_bootstrap, seed=seed,
    )
    print(f"  Demographic AUROC: {auc:.3f} [{lo:.3f}-{hi:.3f}]")
    return {"auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "features": demo_feats}


# =============================================================================
# Run all (public release: manuscript main analyses only)
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Public release: manuscript main analyses only."
    )
    parser.add_argument("--diary-csv", required=True, help="Path to diary_features.csv")
    parser.add_argument("--vocab-csv", required=True, help="Path to lexical/day-level CSV")
    parser.add_argument("--survey-csv", required=True, help="Path to survey CSV")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Run only the main manuscript evaluation without temporal analysis.",
    )
    return parser.parse_args()


def run_all(diary_csv: str, vocab_csv: str, survey_csv: str, out_dir: str,
            seed: int = SEED, run_temporal: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("SEED:", seed)
    print("OUTER:", OUTER_REPEATS, "x", OUTER_SPLITS)
    print("MODEL_WHITELIST:", MODEL_WHITELIST)
    print("[PUBLIC RELEASE] Running manuscript main analyses only: main evaluation + temporal analysis.")

    paths = Paths(diary_csv=diary_csv, vocab_csv=vocab_csv, survey_csv=survey_csv)
    cfg = EvalConfig(
        phq_healthy_max_exclusive=4, phq_depress_min_exclusive=10,
        outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
        inner_splits_thr=INNER_SPLITS_THR, threshold_mode=THR_MODE,
        sens_target=SENS_TARGET, seed=seed, dpi=DPI,
        day_agg=DAY_AGG, prt_agg=PRT_AGG,
        add_acoustic_std=ADD_ACOUSTIC_STD, add_lexical_std=ADD_LEXICAL_STD,
        nested_stable_eval=NESTED_STABLE_EVAL,
        stable_min_rate=STABLE_MIN_RATE,
        stable_fallback_min_rate=STABLE_FALLBACK_MIN_RATE,
        stable_min_features=STABLE_MIN_FEATURES,
        stable_selection_splits=STABLE_SELECTION_SPLITS,
        stable_selection_repeats=STABLE_SELECTION_REPEATS,
        enable_grid_search=ENABLE_GRID_SEARCH,
        grid_search_inner_splits=GRID_SEARCH_INNER_SPLITS,
        grid_search_n_jobs=GRID_SEARCH_N_JOBS,
    )

    df_summary, store, pack = run_main_pipeline(paths, out_dir, cfg, model_whitelist=MODEL_WHITELIST)

    temporal = None
    if run_temporal and pack.get("best_info"):
        temporal = run_temporal_analysis(
            pack["df_main"], pack["acoustic_cols"], pack["lexical_cols"],
            cfg, pack["clf_models"], pack["stable_features"],
            out_dir, max_day=MAX_TEMP_DAY, min_days_per_prt=TEMP_MIN_DAYS_PER_PRT,
            window_sizes=TEMP_WINDOW_SIZES, early_split_day=TEMP_EARLY_SPLIT_DAY,
            stable_nfeat_min=pack.get("stable_nfeat_min"),
            stable_nfeat_max=pack.get("stable_nfeat_max"),
        )

    return df_summary, store, pack, cfg, temporal


if __name__ == "__main__":
    args = parse_args()
    run_all(
        diary_csv=args.diary_csv,
        vocab_csv=args.vocab_csv,
        survey_csv=args.survey_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        run_temporal=not args.skip_temporal,
    )
