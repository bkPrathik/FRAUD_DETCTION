import optuna
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score
from scipy.stats import ks_2samp

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Inner objective (not called directly by user) ─────────────────────────
def _objective(trial, X, y, amounts):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "tree_method":      "hist",
        "device":           "cpu",
        "random_state":     42,
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 150),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucpr_scores, ks_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr,  X_val   = X.iloc[train_idx], X.iloc[val_idx]
        y_tr,  y_val   = y[train_idx], y[val_idx]
        amounts_val    = amounts[val_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=50, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        probs      = model.predict_proba(X_val)[:, 1]
        aucpr      = average_precision_score(y_val, probs, sample_weight=amounts_val)
        aucpr_scores.append(aucpr)

        ks_stat, _ = ks_2samp(probs[y_val == 1], probs[y_val == 0])
        ks_scores.append(ks_stat)

        trial.report(np.mean(aucpr_scores), step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("ks_statistic", np.mean(ks_scores))
    return np.mean(aucpr_scores)


# ── Main function called from other notebooks ─────────────────────────────
def run_fraud_tuning(
    X_train,                     # training features (pandas DataFrame) — holdout excluded externally
    y_train,                     # training labels   (numpy array) — holdout excluded externally
    n_trials     = 4,            # number of Optuna trials
    test_size    = 0.2,          # fraction of X_train held out for internal evaluation
    random_state = 42,
    study_name   = "xgb_fraud_scoring"
):
    """
    Runs Optuna hyperparameter tuning for XGBoost fraud scoring model.
    Optimises for amount-weighted AUC-PR — transactions are weighted by dollar
    amount so the objective penalises missing high-value fraud more heavily.

    NOTE: Holdout data must be separated BEFORE calling this function.
    Only pass the training portion here — this function has no knowledge
    of holdout and cannot leak it.

    Args:
        X_train      : Feature matrix (pandas DataFrame), holdout already removed.
                       Must contain an 'amount' column for weighted AUC-PR.
        y_train      : Label vector   (numpy array), holdout already removed
        n_trials     : Number of Optuna trials (default 4)
        test_size    : Fraction of X_train used for internal test eval (default 0.2)
        random_state : Reproducibility seed (default 42)
        study_name   : Name of Optuna study (default 'xgb_fraud_scoring')

    Returns:
        dict with keys:
            - 'model'       : final XGBClassifier trained on full X_train
            - 'study'       : Optuna study object
            - 'best_params' : best hyperparameters found
            - 'cv_aucpr'    : mean amount-weighted AUC-PR across CV folds (best trial)
            - 'cv_ks'       : mean KS statistic across CV folds (best trial)
            - 'test_aucpr'  : amount-weighted AUC-PR on internal test split
            - 'test_ks'     : KS statistic on internal test split
    """

    # ── FIX: convert y to numpy so train_test_split outputs are index-free ─
    if hasattr(y_train, "values"):
        y_train = y_train.values

    # ── Extract transaction amounts for weighted AUC-PR ───────────────────
    if hasattr(X_train, "columns") and "amount" in X_train.columns:
        amounts = X_train["amount"].values
    else:
        amounts = np.ones(len(y_train))  # fallback: uniform weights

    # ── 1. Internal train / test split (for tuning evaluation only) ───────
    X_tr, X_te, y_tr, y_te, amounts_tr, amounts_te = train_test_split(
        X_train, y_train, amounts,
        test_size    = test_size,
        stratify     = y_train,
        random_state = random_state
    )

    # ── 2. Run Optuna study on X_tr ───────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = study_name
    )

    study.optimize(
        lambda trial: _objective(trial, X_tr, y_tr, amounts_tr),
        n_trials          = n_trials,
        show_progress_bar = True
    )

    # ── 3. Evaluate best params on internal test split ────────────────────
    best_params  = study.best_params
    eval_model   = xgb.XGBClassifier(
        **best_params,
        objective    = "binary:logistic",
        eval_metric  = "aucpr",
        random_state = random_state
    )
    eval_model.fit(X_tr, y_tr)

    test_probs  = eval_model.predict_proba(X_te)[:, 1]
    test_aucpr  = average_precision_score(y_te, test_probs, sample_weight=amounts_te)
    test_ks, _  = ks_2samp(test_probs[y_te == 1], test_probs[y_te == 0])

    # ── 4. Train final model on full X_train (no data left out) ──────────
    final_model = xgb.XGBClassifier(
        **best_params,
        objective    = "binary:logistic",
        eval_metric  = "aucpr",
        random_state = random_state
    )
    final_model.fit(X_train, y_train)

    # ── 5. Print summary ──────────────────────────────────────────────────
    print("=" * 50)
    print(f"  CV   Wtd AUC-PR (best trial) : {study.best_trial.value:.4f}")
    print(f"  CV   KS Stat    (best trial) : {study.best_trial.user_attrs['ks_statistic']:.4f}")
    print(f"  Internal Test Wtd AUC-PR     : {test_aucpr:.4f}")
    print(f"  Internal Test KS Stat        : {test_ks:.4f}")
    print(f"  Best Params                  : {best_params}")
    print("=" * 50)
    print("  Final model retrained on full X_train.")
    print("  Apply to holdout externally in your prediction notebook.")
    print("=" * 50)

    return {
        "model":       final_model,
        "study":       study,
        "best_params": best_params,
        "cv_aucpr":    study.best_trial.value,
        "cv_ks":       study.best_trial.user_attrs["ks_statistic"],
        "test_aucpr":  test_aucpr,
        "test_ks":     test_ks,
    }