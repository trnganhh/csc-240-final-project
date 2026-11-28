"""
Decision Tree + Random Forest (Model 3) for preterm birth.

Questions answered
  H5  Does a more complex algorithm (tree / forest) beat logistic regression
      on the same features, and does it hold up on a new year (2025)?
  H6  Does adding total prenatal visits (PREVIS, leakage-prone) inflate
      apparent performance?

Design
  train : random sample of 2024           (fit the models)
  valid : separate random sample of 2024  (tune hyper-parameters + threshold)
  test  : ALL of 2025                     (temporal validation, touched once)

Run:    python tree_models.py            (full run, ~20-40 min on 6 cores)
        python tree_models.py --quick    (small samples, ~2 min, for testing)
Output: results/tree_models/
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 240
OUT = Path("results/tree_models")

# 1.  FEATURES  (same list should be used by the full logistic regression)
# Only things known BEFORE or EARLY in pregnancy. Left out on purpose:
#   - gestational diabetes / hypertension / eclampsia: diagnosed during the
#     pregnancy and are often the *reason* for an early delivery
#   - prenatal_start_month: leaks (a 32-week birth can't start care in month 9);
#     prenatal_first_tri / no_prenatal_care are used instead
#   - prenatal_visits, weight_gain: grow with pregnancy length -> H6 only
#   - birth weight, baby sex, gestation: outcomes of the birth
NUMERIC = [
    "mother_age", "bmi", "height_in", "education", "married",
    "smoked_before", "smoked_during", "cigs_before",
    "prior_births", "prior_deaths", "prior_terminations", "first_baby",
    "months_since_last_birth",
    "diabetes_prepreg", "hypertension_prepreg", "prior_preterm",
    "prior_cesarean", "infertility_treatment",
    "prenatal_first_tri", "no_prenatal_care",
]
RACE_CODES = {1: "nh_white", 2: "nh_black", 3: "nh_aian", 4: "nh_asian",
              5: "nh_nhopi", 6: "nh_multi", 7: "hispanic"}
LEAKY = "prenatal_visits"
TARGET = "preterm"


def risk_count(df):
    """Engineered feature (proposal: 'cumulative risk indicators'): how many
    of 6 established, pre-pregnancy risk factors the mother has. Built only
    from leakage-free variables; an unknown factor counts as 0."""
    flags = [
        (df["mother_age"] < 20) | (df["mother_age"] >= 35),  # age extremes
        df["bmi"] >= 30,                                      # obesity
        df["smoked_before"] == 1,
        df["hypertension_prepreg"] == 1,
        df["diabetes_prepreg"] == 1,
        df["prior_preterm"] == 1,
    ]
    return sum(f.fillna(False).astype("float32") for f in flags)


def build_X(df, with_leak=False):
    """Numeric matrix. Missing values stay NaN: sklearn >= 1.4 trees split
    on missingness directly, so no imputation is needed for DT / RF."""
    X = df[NUMERIC].astype("float32")
    X["risk_count"] = risk_count(df)
    for code, name in RACE_CODES.items():
        X[f"race_{name}"] = (df["race_hispanic"] == code).astype("float32")
        X.loc[df["race_hispanic"].isna(), f"race_{name}"] = np.nan
    if with_leak:
        X[LEAKY] = df[LEAKY].astype("float32")
    return X


def load(args):
    cols = NUMERIC + ["race_hispanic", LEAKY, TARGET]
    d24 = pd.read_parquet("clean/natality_2024.parquet", columns=cols)
    d25 = pd.read_parquet("clean/natality_2025.parquet", columns=cols)
    shuffled = d24.sample(frac=1.0, random_state=SEED)
    train = shuffled.iloc[:args.n_train]
    valid = shuffled.iloc[args.n_train:args.n_train + args.n_valid]
    test = d25 if args.n_test is None else d25.sample(args.n_test, random_state=SEED)
    print(f"train {len(train):,} (2024)  valid {len(valid):,} (2024)  "
          f"test {len(test):,} (2025)")
    print(f"preterm rate  train {train[TARGET].mean():.2%}  "
          f"valid {valid[TARGET].mean():.2%}  test {test[TARGET].mean():.2%}")
    return train, valid, test


# 2.  METRICS
# Preterm is ~9% of births, so accuracy is useless (predict "term" = 91%).
# ROC-AUC and PR-AUC don't need a threshold. For precision / recall / F1 we
# pick the threshold that maximises F1 on the 2024 VALIDATION set and then
# apply that same threshold to 2025 (choosing it on 2025 would be cheating).

def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def scores(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat),
            "threshold": thr}


# 3.  TUNING  (on the 2024 validation set, by ROC-AUC)

def tune(make, grid, Xtr, ytr, Xva, yva, label):
    best, best_auc, log = None, -1, []
    for params in grid:
        t = time.time()
        m = make(**params).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        log.append({**params, "valid_auc": round(auc, 4)})
        print(f"  {label} {params}  valid AUC {auc:.4f}  ({time.time()-t:.0f}s)",
              flush=True)
        if auc > best_auc:
            best, best_auc = params, auc
    return best, log


def dt(**p):
    return DecisionTreeClassifier(random_state=SEED, **p)


def rf(n_estimators=200, **p):
    return RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt",
                                  max_samples=0.5, n_jobs=-1,
                                  random_state=SEED, **p)


DT_GRID = [{"max_depth": d, "min_samples_leaf": l}
           for d in (4, 6, 8, 10, 14) for l in (100, 500)]
RF_GRID = [{"max_depth": d, "min_samples_leaf": l, "n_estimators": 100}
           for d in (10, 16, None) for l in (50, 200)]


# 4.  MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_valid, args.n_test = 60_000, 30_000, 100_000
    else:
        args.n_train, args.n_valid, args.n_test = 800_000, 300_000, None
    OUT.mkdir(parents=True, exist_ok=True)

    train, valid, test = load(args)
    ytr, yva, yte = (d[TARGET].to_numpy(int) for d in (train, valid, test))

    rows, curves, tuning, models = [], {}, {}, {}
    for leak in (False, True):
        tag = "+PREVIS" if leak else ""
        Xtr, Xva, Xte = (build_X(d, leak) for d in (train, valid, test))

        candidates = {}
        if not leak:
            # reference baseline (Navya owns the official Model 1 / Model 2)
            age = make_pipeline(StandardScaler(), LogisticRegression())
            candidates["Age-only LR (baseline)"] = age.fit(
                Xtr[["mother_age"]], ytr)

        print(f"\nTuning decision tree {tag}")
        p, tuning[f"decision_tree{tag}"] = tune(dt, DT_GRID, Xtr, ytr, Xva, yva, "DT")
        candidates[f"Decision tree{tag}"] = dt(**p).fit(Xtr, ytr)

        print(f"\nTuning random forest {tag}")
        p, tuning[f"random_forest{tag}"] = tune(rf, RF_GRID, Xtr, ytr, Xva, yva, "RF")
        p = {**p, "n_estimators": 300}           # final model: more trees
        t = time.time()
        candidates[f"Random forest{tag}"] = rf(**p).fit(Xtr, ytr)
        print(f"  final RF {p} fit in {time.time()-t:.0f}s")

        for name, m in candidates.items():
            cols = ["mother_age"] if name.startswith("Age") else list(Xtr.columns)
            pva = m.predict_proba(Xva[cols])[:, 1]
            pte = m.predict_proba(Xte[cols])[:, 1]
            thr = best_f1_threshold(yva, pva)
            for split, y, pr in (("valid_2024", yva, pva), ("test_2025", yte, pte)):
                rows.append({"model": name, "split": split, "leakage_feature": leak,
                             **scores(y, pr, thr)})
            curves[name] = pte
            models[name] = (m, cols)
            print(f"  {name:28s} 2024-valid AUC {rows[-2]['roc_auc']:.4f}  "
                  f"2025-test AUC {rows[-1]['roc_auc']:.4f}")

    # ---- tables
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    wide = res.pivot_table(index="model", columns="split",
                           values=["roc_auc", "pr_auc", "f1"])
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide["auc_drop_2024_to_2025"] = wide["roc_auc_valid_2024"] - wide["roc_auc_test_2025"]
    order = [n for n in curves]
    wide = wide.loc[order].round(4)
    wide.to_csv(OUT / "summary.csv")
    with open(OUT / "tuning.json", "w") as f:
        json.dump(tuning, f, indent=2, default=str)
    print("\n", wide.to_string())

    # ---- H6 leakage table
    t25 = res[res.split == "test_2025"].set_index("model")
    leak_tab = pd.DataFrame({
        algo: {"auc_without": t25.loc[algo, "roc_auc"],
               "auc_with_PREVIS": t25.loc[f"{algo}+PREVIS", "roc_auc"],
               "pr_auc_without": t25.loc[algo, "pr_auc"],
               "pr_auc_with_PREVIS": t25.loc[f"{algo}+PREVIS", "pr_auc"]}
        for algo in ("Decision tree", "Random forest")}).T
    leak_tab["auc_gain"] = leak_tab.auc_with_PREVIS - leak_tab.auc_without
    leak_tab.round(4).to_csv(OUT / "leakage_h6.csv")
    print("\nH6 leakage (2025 test)\n", leak_tab.round(4).to_string())

    # ---- figures
    plot_curves(curves, yte)
    plot_importance(models, test, yte, args)
    plot_small_tree(train, ytr)
    print(f"\nAll outputs in {OUT.resolve()}")


# 5.  FIGURES

COLORS = {"Age-only LR (baseline)": "#666666", "Decision tree": "#f47230",
          "Random forest": "#257de0"}


def style(name):
    base = name.replace("+PREVIS", "")
    return dict(color=COLORS[base], lw=2,
                ls="--" if name.endswith("+PREVIS") else "-")


def plot_curves(curves, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name}  (AUC {roc_auc_score(y, p):.3f})",
                     **style(name))
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name}  (AP {average_precision_score(y, p):.3f})",
                     **style(name))
    axes[0].plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    axes[1].axhline(y.mean(), color="#bbbbbb", lw=1)
    axes[0].set(title="ROC curve - 2025 test set", xlabel="False positive rate",
                ylabel="True positive rate")
    axes[1].set(title="Precision-recall curve - 2025 test set", xlabel="Recall",
                ylabel="Precision")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, -0.02, "Dashed = model also given total prenatal visits "
             "(leakage-prone, H6)", ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(models, test, y, args):
    """Permutation importance on the 2025 test set: how much ROC-AUC drops
    when one feature is shuffled. Unlike the built-in impurity importance,
    this is not biased toward continuous features like age / BMI."""
    n = 20_000 if args.quick else 60_000
    idx = np.random.default_rng(SEED).choice(len(test), n, replace=False)
    sub, ysub = test.iloc[idx], y[idx]
    m, cols = models["Random forest"]
    X = build_X(sub)[cols]
    print("\nPermutation importance (random forest) ...", flush=True)
    r = permutation_importance(m, X, ysub, scoring="roc_auc", n_repeats=3,
                               random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "auc_drop": r.importances_mean,
                        "std": r.importances_std,
                        "impurity_importance": m.feature_importances_})
    imp = imp.sort_values("auc_drop", ascending=False)
    imp.to_csv(OUT / "feature_importance_rf.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.feature, top.auc_drop, xerr=top["std"], color="#257de0",
            height=0.6, error_kw={"ecolor": "#666666", "lw": 1})
    ax.set(title="Random forest: permutation importance (2025 test)",
           xlabel="Drop in ROC-AUC when the feature is shuffled")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance_rf.png", dpi=150)
    plt.close(fig)


def plot_small_tree(train, y):
    """A depth-3 tree is too weak to compete, but it is readable: it shows
    which splits the data finds first. Good for the slides."""
    X = build_X(train)
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
                               random_state=SEED).fit(X, y)
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(m, feature_names=list(X.columns), class_names=["term", "preterm"],
              filled=True, proportion=True, impurity=False, rounded=True,
              fontsize=9, ax=ax)
    ax.set_title("Depth-3 decision tree (2024 training data) - "
                 "'value' = [share term, share preterm]")
    fig.savefig(OUT / "decision_tree_depth3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
