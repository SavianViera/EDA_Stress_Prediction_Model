"""
Leave-One-Subject-Out (LOSO) Evaluation Pipeline (Birkenmaier et al., 2026)
=============================================================================
Matches the paper's evaluation protocol: for each of N subjects, hold that
subject out entirely, train everything else on the remaining N-1 subjects,
and evaluate once on the held-out subject. Repeat for every subject, then
report mean +/- SD across all folds (paper's Table 5 format).

Two-phase design, matching the paper's own description ("the search was
conducted on the complete dataset prior to LOSO evaluation... the best
parameters were stored and used to instantiate fresh classifier objects
within each LOSO fold"):

  PHASE 1 (once): Train a REFERENCE autoencoder on ALL subjects, purely to
  generate a feature set for the classifier hyperparameter search (Table 4
  grids, subject-grouped inner CV). This reference model is used ONLY to
  pick hyperparameters -- it never touches the actual LOSO evaluation.

  PHASE 2 (N times): The real LOSO loop. Each fold trains a FRESH
  autoencoder from scratch on N-1 subjects (the held-out subject is
  completely invisible to it), extracts that fold's own latent features,
  fuses with the (subject-independent, precomputed once) statistical
  features, fits classifiers using the FIXED hyperparameters from Phase 1,
  and evaluates once on the held-out subject.

COST WARNING: this trains N full autoencoders (default 15 subjects x
FINAL_EPOCHS each) plus N x len(CLASSIFIER_GRIDS) classifier fits. See
QUICK_TEST_SUBJECTS / LOSO_EPOCHS below to control runtime.
"""

import time

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from eda_hybrid_pipeline import compute_statistical_features, CLASSIFIER_GRIDS
from eda_statistical_features import normalize_features_per_subject
from eda_supervised_autoencoder import (
    SEED,
    FINAL_EPOCHS,
    _find_and_prepare_wesad_root,
    load_wesad_eda_windows,
    train_supervised_autoencoder_final,
)

LAMBDA_CLS = 2.0
N_INNER_FOLDS = 5           # inner CV folds for the Phase-1 hyperparameter search
LOSO_EPOCHS = FINAL_EPOCHS  # autoencoder epochs PER LOSO FOLD -- lower this for a quick test
REFERENCE_EPOCHS = FINAL_EPOCHS  # epochs for the Phase-1 reference autoencoder

# Winning combination from eda_cvxeda_search.py (54-combo grid, CV Acc 0.9106,
# best of all combinations tried). This is the SINGLE SOURCE OF TRUTH for
# these parameters in this script -- passed explicitly at the
# compute_statistical_features call site below, not relying on that
# function's own internal defaults (which are still the old unadapted
# values, and will silently override anything you edit elsewhere if this
# dict isn't passed explicitly).
CVXEDA_PARAMS = dict(tau0=1.0, tau1=0.7, delta_knot=40.0, alpha=4e-4)

# Winning cutoff from eda_highpass_search.py: beat the paper's literal
# 0.05 Hz on every tracked metric simultaneously (S17, S7, and overall
# accuracy) -- see search results for the full trade-off analysis.
HIGHPASS_CUTOFF_HZ = 0.03

# Set to a small int (e.g. 3) to run LOSO on only the first N subjects, for
# a fast sanity check before committing to the full run. None = all subjects.
QUICK_TEST_SUBJECTS = None


def search_best_hyperparams(name, estimator, param_grid, X, y, groups,
                             n_folds=N_INNER_FOLDS, seed=SEED):
    """Same search as eda_hybrid_pipeline.tune_and_evaluate, but returns only
    the winning hyperparameters (no final refit/evaluate -- that happens
    per-fold in the LOSO loop instead)."""
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)

    n_unique_groups = len(set(groups.tolist() if hasattr(groups, "tolist") else groups))
    effective_folds = min(n_folds, n_unique_groups)

    inner_cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, param_grid, scoring="accuracy", cv=inner_cv,
                           n_jobs=-1, refit=False)

    print(f"\n{name}: searching {n_combos} combos x {effective_folds} folds "
          f"= {n_combos * effective_folds} fits...")
    start = time.time()
    search.fit(X, y, groups=groups)
    elapsed = time.time() - start

    print(f"  Done in {elapsed:.1f}s | Best params: {search.best_params_} | "
          f"CV Acc: {search.best_score_:.4f}")
    return search.best_params_


def run_loso(windows, subject_ids, labels, raw_win_len, stat_features,
             fixed_hyperparams, device, lambda_cls=LAMBDA_CLS, ae_epochs=LOSO_EPOCHS,
             seed=SEED, quick_test_subjects=QUICK_TEST_SUBJECTS):
    """
    The real LOSO loop. `fixed_hyperparams` is a dict: name -> (estimator, params),
    as produced by Phase 1. Returns per-fold, per-classifier results.
    """
    all_subjects = sorted(set(subject_ids.tolist()))
    if quick_test_subjects is not None:
        all_subjects = all_subjects[:quick_test_subjects]
        print(f"\n*** QUICK TEST MODE: running LOSO on only {len(all_subjects)} "
              f"subjects: {all_subjects} ***")

    fold_results = {name: [] for name in fixed_hyperparams}

    for fold_idx, test_subject in enumerate(all_subjects, 1):
        train_subjects = [s for s in set(subject_ids.tolist()) if s != test_subject]
        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = subject_ids == test_subject

        train_windows = windows[torch.from_numpy(train_mask)]
        test_windows = windows[torch.from_numpy(test_mask)]
        train_labels_t = torch.from_numpy(labels[train_mask])
        test_labels_t = torch.from_numpy(labels[test_mask])

        print(f"\n{'=' * 70}")
        print(f"LOSO FOLD {fold_idx}/{len(all_subjects)} | Held-out subject: {test_subject} "
              f"({int(test_mask.sum())} windows)")
        print(f"{'=' * 70}")

        # Fresh, held-out-subject-blind autoencoder for this fold
        model, history = train_supervised_autoencoder_final(
            train_windows, train_labels_t, test_windows, test_labels_t,
            raw_win_len, device, lambda_cls=lambda_cls, epochs=ae_epochs,
            seed=seed, verbose=False,
        )
        print(f"  Autoencoder: Test MSE={history['test_recon_mse'][-1]:.6f}, "
              f"Test Acc (latent-only)={history['test_acc'][-1]:.4f}")

        model.eval()
        with torch.no_grad():
            latent_features = model.encoder(windows.to(device)).cpu().numpy()

        hybrid_features = np.concatenate([stat_features, latent_features], axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(hybrid_features[train_mask])
        X_test = scaler.transform(hybrid_features[test_mask])
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        for name, (estimator, params) in fixed_hyperparams.items():
            clf = clone(estimator).set_params(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            fold_results[name].append({
                "subject": test_subject, "accuracy": acc, "f1": f1,
                "sensitivity": sens, "specificity": spec,
            })
            print(f"  {name:<15} Acc={acc:.4f} F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f}")

    return fold_results


def summarize_loso(fold_results):
    print("\n" + "=" * 70)
    print(f"LOSO SUMMARY (mean +/- SD over {len(next(iter(fold_results.values())))} folds)")
    print("=" * 70)

    summary = {}
    for name, folds in sorted(fold_results.items(), key=lambda kv: -np.mean([f["accuracy"] for f in kv[1]])):
        accs = np.array([f["accuracy"] for f in folds])
        f1s = np.array([f["f1"] for f in folds])
        sens = np.array([f["sensitivity"] for f in folds])
        spec = np.array([f["specificity"] for f in folds])
        print(f"{name:<15} Acc={accs.mean():.4f}+/-{accs.std():.4f}  "
              f"F1={f1s.mean():.4f}+/-{f1s.std():.4f}  "
              f"Sens={sens.mean():.4f}+/-{sens.std():.4f}  "
              f"Spec={spec.mean():.4f}+/-{spec.std():.4f}")
        summary[name] = {
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "sens_mean": float(sens.mean()), "sens_std": float(sens.std()),
            "spec_mean": float(spec.mean()), "spec_std": float(spec.std()),
        }

    print("\nPer-subject accuracy (spot the hard subjects):")
    for name, folds in fold_results.items():
        per_subj = ", ".join(f"{f['subject']}={f['accuracy']:.2f}" for f in folds)
        print(f"  {name:<15} {per_subj}")

    return summary


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wesad_root = _find_and_prepare_wesad_root()
    print(f"Using WESAD_ROOT = {wesad_root}")

    windows, subject_ids, labels, raw_win_len, starts = load_wesad_eda_windows(
        wesad_root, window_sec=60.0, stride_sec=30.0, target_len=256, fs=4,
        apply_highpass=True, highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
    )

    # Statistical features are subject-independent (cvxEDA decomposed per
    # subject, no cross-subject info) -- safe to compute ONCE, reused by
    # every LOSO fold.
    print("\n" + "=" * 70)
    print("Computing statistical features once (Table 3, subject-independent)")
    print("=" * 70)
    stat_features = compute_statistical_features(wesad_root, subject_ids, starts, raw_win_len,
                                                  fs=4, apply_highpass=True,
                                                  highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
                                                  **CVXEDA_PARAMS)
    print(f"Statistical feature matrix: {stat_features.shape}")

    # Per-subject normalization -- removes between-subject differences in
    # absolute EDA reactivity scale (a subject can have a completely normal
    # within-subject stress>baseline direction while still being
    # misclassified because their absolute amplitudes are much smaller than
    # other subjects'). Still subject-only, still safe to apply once
    # upfront -- no leakage across LOSO folds.
    stat_features = normalize_features_per_subject(stat_features, subject_ids)
    print("Applied per-subject normalization to statistical features")

    # ============================================================
    # PHASE 1: fix classifier hyperparameters (once, reference model)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Reference autoencoder (all subjects) -> fix hyperparameters")
    print("=" * 70)
    all_subjects_list = sorted(set(subject_ids.tolist()))
    dummy_test_subject = all_subjects_list[0]  # trainer needs SOME held-out split;
    ref_train_mask = subject_ids != dummy_test_subject
    ref_test_mask = subject_ids == dummy_test_subject

    ref_model, ref_history = train_supervised_autoencoder_final(
        windows[torch.from_numpy(ref_train_mask)], torch.from_numpy(labels[ref_train_mask]),
        windows[torch.from_numpy(ref_test_mask)], torch.from_numpy(labels[ref_test_mask]),
        raw_win_len, device, lambda_cls=LAMBDA_CLS, epochs=REFERENCE_EPOCHS,
        seed=SEED, verbose=False,
    )
    ref_model.eval()
    with torch.no_grad():
        ref_latent = ref_model.encoder(windows.to(device)).cpu().numpy()
    ref_hybrid = np.concatenate([stat_features, ref_latent], axis=1)

    ref_scaler = StandardScaler()
    X_ref = ref_scaler.fit_transform(ref_hybrid)  # all subjects -- reference-only, never used for LOSO metrics
    y_ref = labels
    groups_ref = subject_ids

    fixed_hyperparams = {}
    for name, cfg in CLASSIFIER_GRIDS.items():
        best_params = search_best_hyperparams(
            name, cfg["estimator"], cfg["param_grid"], X_ref, y_ref, groups_ref
        )
        fixed_hyperparams[name] = (cfg["estimator"], best_params)

    # ---- Build ensembles (Eq. 2) from each base classifier's OWN tuned
    # hyperparameters found above -- not re-tuned as an ensemble, matching
    # the paper's approach. clone() on a VotingClassifier recursively
    # clones its named sub-estimators too, so pairing these with an empty
    # params dict lets them reuse run_loso's existing
    # clone(estimator).set_params(**params) loop unmodified below.
    rf_est = clone(fixed_hyperparams["Random Forest"][0]).set_params(**fixed_hyperparams["Random Forest"][1])
    xgb_est = clone(fixed_hyperparams["XGBoost"][0]).set_params(**fixed_hyperparams["XGBoost"][1])
    svm_est = clone(fixed_hyperparams["SVM"][0]).set_params(**fixed_hyperparams["SVM"][1])
    knn_est = clone(fixed_hyperparams["KNN"][0]).set_params(**fixed_hyperparams["KNN"][1])

    ensemble_rf_xgb = VotingClassifier(
        estimators=[("rf", rf_est), ("xgb", xgb_est)], voting="soft", weights=[1, 1],
    )
    ensemble_all = VotingClassifier(
        estimators=[("rf", rf_est), ("xgb", xgb_est), ("svm", svm_est), ("knn", knn_est)],
        voting="soft", weights=[2, 2, 1, 1],
    )
    fixed_hyperparams["Ensemble_RF_XGB"] = (ensemble_rf_xgb, {})
    fixed_hyperparams["Ensemble_All"] = (ensemble_all, {})
    print(f"\nBuilt Ensemble_RF_XGB (equal weights) and Ensemble_All "
          f"(RF/XGB x2, SVM/KNN x1) from each classifier's own tuned hyperparameters.")

    # ============================================================
    # PHASE 2: the real LOSO loop
    # ============================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: LOSO evaluation ({len(all_subjects_list)} subjects, "
          f"{LOSO_EPOCHS} epochs/fold autoencoder)")
    print("=" * 70)

    fold_results = run_loso(
        windows, subject_ids, labels, raw_win_len, stat_features,
        fixed_hyperparams, device, lambda_cls=LAMBDA_CLS, ae_epochs=LOSO_EPOCHS,
    )

    summary = summarize_loso(fold_results)