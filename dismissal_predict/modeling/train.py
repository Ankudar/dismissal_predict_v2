from collections import Counter
from datetime import datetime
from functools import partial
from itertools import product
import json
import logging
import os
import warnings

import joblib
import lightgbm as lgb
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import log_evaluation
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# для наглядности лучше или хуже новая модель
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
MODELS = "/home/root6/python/dismissal_predict_v2/models"
os.makedirs(MODELS, exist_ok=True)

INPUT_FILE_MAIN_USERS = f"{DATA_PROCESSED}/main_users_for_train.csv"
INPUT_FILE_TOP_USERS = f"{DATA_PROCESSED}/main_top_for_train.csv"

TARGET_COL = "уволен"

TEST_SIZE = 0.25
RANDOM_STATE = 40
N_TRIALS = 10000  # итерации для оптуны
N_SPLITS = 5  # число кроссвалидаций
METRIC = "f2"
MLFLOW_EXPERIMENT_MAIN = "main_users"
MLFLOW_EXPERIMENT_TOP = "top_users"
EARLY_STOP = 500
BETA_FOR_F2 = 20


N_JOBS = -1
THRESHOLDS = np.arange(0.1, 0.9, 0.01)
MIN_PRECISION = 0.4  # в текущей версии не используется.. удалить или реализовать

# сетка для перебора
# перебор FN_WEIGHT и FP_WEIGHT - FN больше значит больше штраф за FN, с FP также
FN_WEIGHT_GRID = range(20, 30)
FP_WEIGHT_GRID = range(0, 5)
N_ITERATIONS = len(FN_WEIGHT_GRID) * len(FP_WEIGHT_GRID)

N_MLFLOW_COUNT = 20  # сколько храним моделей по убыванию ключевой метрики

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")


class EarlyStoppingCallback:
    def __init__(self, patience=EARLY_STOP, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        if study.best_value > self.best_score + self.min_delta:
            self.best_score = study.best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
            logger.info(f"Ранняя остановка: нет улучшений {self.patience} trials")


def custom_metric_from_counts(tp, tn, fn, fp, beta=BETA_FOR_F2, fn_tolerance=2):
    # Если FN больше допустимого порога → метрика 0
    if fn > fn_tolerance:
        return 0.0

    # Precision и Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Если и precision и recall = 0 → метрика 0
    if precision + recall == 0:
        return 0.0

    # F-beta (по умолчанию F2)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return round(f_beta, 6)


def get_confusion_counts(cm):
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    return tn, fp, fn, tp


def is_new_model_better(new_metrics, old_metrics, delta=0.001):
    def round3(x):
        return round(x or 0, 3)

    new_score = round3(new_metrics.get(METRIC, 0))
    old_score = round3(old_metrics.get(METRIC, 0))

    if new_score > old_score + delta:
        return True
    if new_score < old_score - delta:
        return False

    return False


def manual_optuna_progress(study, n_trials, func):
    for _ in tqdm(range(n_trials), desc="Optuna Tuning"):
        study.optimize(func, n_trials=1, catch=(Exception,))


def convert_all_to_float(df: pd.DataFrame, exclude_cols=None):
    df = df.copy()
    exclude_cols = exclude_cols or []

    for col in df.columns:
        if col in exclude_cols:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
            .replace(["nan", "None", "", "NaN"], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_metrics(y_true, y_proba, threshold, beta=BETA_FOR_F2):
    """Считает метрики по заданному трешхолду."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = get_confusion_counts(cm)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "custom": custom_metric_from_counts(tp, tn, fn, fp, beta=beta),
    }


def find_best_threshold_weighted(y_true, y_proba, thresholds, fn_weight=2.0, fp_weight=10.0):
    """
    Подбирает трешхолд по минимизации fn_weight*FN + fp_weight*FP.
    Приоритет у FN, но учитывается и FP.
    """
    best_thr, best_score = 0.5, float("inf")
    best_fn, best_fp = float("inf"), float("inf")

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = get_confusion_counts(confusion_matrix(y_true, y_pred, labels=[0, 1]))
        score = fn_weight * fn + fp_weight * fp

        # ищем минимум score, при равенстве смотрим на FN/FP
        if score < best_score or (
            score == best_score and (fn < best_fn or (fn == best_fn and fp < best_fp))
        ):
            best_thr, best_score, best_fn, best_fp = t, score, fn, fp

    return best_thr, best_fn, best_fp


def hybrid_objective(
    trial,
    X_train,
    y_train,
    fn_weight=10,
    fp_weight=2,
    n_iter=None,
):
    """Objective-функция для Optuna: SelectKBest один раз + LightGBM + CV + early stopping + pruner."""
    try:
        # --- feature selection ---
        k_best = trial.suggest_int("k_best", 5, min(40, X_train.shape[1]))
        feature_names = X_train.columns.tolist()

        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_transformed = selector.fit_transform(X_train, y_train)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

        # --- hyperparams ---
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
            "n_jobs": 2,
            "verbose": -1,
        }

        # --- CV ---
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_transformed, y_train)):
            X_tr, X_val = X_train_transformed[train_idx], X_train_transformed[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            y_val_proba = model.predict_proba(X_val)[:, 1]

            # --- подбор оптимального трешхолда ---
            thresholds = THRESHOLDS
            best_threshold, best_fn, best_fp = find_best_threshold_weighted(
                y_val, y_val_proba, thresholds=THRESHOLDS, fn_weight=fn_weight, fp_weight=fp_weight
            )

            # --- расчет метрики на оптимальном трешхолде ---
            if METRIC == "f1":
                score = f1_score(y_val, (y_val_proba >= best_threshold).astype(int))
            elif METRIC == "accuracy":
                score = accuracy_score(y_val, (y_val_proba >= best_threshold).astype(int))
            elif METRIC == "recall":
                score = recall_score(y_val, (y_val_proba >= best_threshold).astype(int))
            elif METRIC == "precision":
                score = precision_score(
                    y_val, (y_val_proba >= best_threshold).astype(int), zero_division=0
                )
            elif METRIC == "roc_auc":
                score = roc_auc_score(y_val, y_val_proba)
            elif METRIC == "custom":
                tn, fp, fn, tp = get_confusion_counts(
                    confusion_matrix(
                        y_val, (y_val_proba >= best_threshold).astype(int), labels=[0, 1]
                    )
                )
                score = custom_metric_from_counts(tp, tn, fn, fp)
            elif METRIC == "f2":
                score = fbeta_score(
                    y_val,
                    (y_val_proba >= best_threshold).astype(int),
                    beta=BETA_FOR_F2,
                    zero_division=0,
                )
            else:
                score = recall_score(y_val, (y_val_proba >= best_threshold).astype(int))

            scores.append(score)

            # --- pruner ---
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = np.mean(scores)
        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("n_selected_features", len(selected_features))

        # --- кастомный вывод каждые N итераций ---
        if trial.number % 50 == 0:
            logger.info(
                f"[{n_iter}/{N_ITERATIONS}] Trial {trial.number}: mean_score={mean_score:.4f} ({METRIC}), "
                f"features={len(selected_features)}, params={trial.params}"
            )

        return mean_score if np.isfinite(mean_score) else -1

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.warning(f"Ошибка в hybrid_objective: {e}")
        return -1


def run_optuna_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    metric,
    n_trials,
    experiment_name,
    model_output_path,
    current_time,
    fn_weight=10,
    fp_weight=2,
    n_iter=None,
):
    """Запуск Optuna + обучение финальной модели + логирование."""
    try:
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        objective = partial(
            hybrid_objective,
            X_train=X_train,
            y_train=y_train,
            fn_weight=fn_weight,
            fp_weight=fp_weight,
            n_iter=n_iter,
        )
        study = optuna.create_study(
            study_name=experiment_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
        early_stopping = EarlyStoppingCallback(patience=EARLY_STOP, min_delta=0.001)

        # запускаем оптимизацию
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[early_stopping],
        )

        # выбранные признаки
        selected_features = study.best_trial.user_attrs.get("selected_features", [])
        X_train_sel, X_test_sel = X_train[selected_features], X_test[selected_features]

        best_params = study.best_trial.params.copy()
        best_params.pop("k_best", None)
        model_params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}

        # финальная модель
        final_model = lgb.LGBMClassifier(**model_params)

        # разбиваем train на train/val для подбора threshold
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sel, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train
        )

        final_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb_early_stopping(stopping_rounds=EARLY_STOP),
                log_evaluation(-1),
            ],
        )

        # подбор threshold на валидации (через find_best_threshold_weighted)
        y_val_proba = final_model.predict_proba(X_val)[:, 1]
        best_threshold, best_fn, best_fp = find_best_threshold_weighted(
            y_val, y_val_proba, thresholds=THRESHOLDS, fn_weight=fn_weight, fp_weight=fp_weight
        )
        logger.info(f"Best threshold: {best_threshold}, FN={best_fn}, FP={best_fp}")

        # --- метрики test/train ---
        y_test_proba = final_model.predict_proba(X_test_sel)[:, 1]
        final_metrics = compute_metrics(y_test, y_test_proba, best_threshold)

        y_train_proba = final_model.predict_proba(X_train_sel)[:, 1]
        final_metrics_train = compute_metrics(y_train, y_train_proba, best_threshold)

        input_example = pd.DataFrame(X_test_sel[:1], columns=X_test_sel.columns)
        run_name = f"model_{current_time}"

        # --- сравнение с локальной моделью ---
        save_model, is_best = True, False
        if os.path.exists(model_output_path):
            try:
                old_model_bundle = joblib.load(model_output_path)
                old_metrics = old_model_bundle.get("metrics", {})

                logger.info(
                    "Старая модель: "
                    + ", ".join(
                        f"{k}: {GREEN + str(v) + RESET}" if k == metric else f"{k}: {v}"
                        for k, v in old_metrics.items()
                    )
                )
                logger.info(
                    "Новая модель: "
                    + ", ".join(
                        f"{k}: {GREEN + str(v) + RESET}" if k == metric else f"{k}: {v}"
                        for k, v in final_metrics.items()
                    )
                )

                save_model = is_new_model_better(final_metrics, old_metrics)
                is_best = save_model

                if save_model:
                    logger.info(f"{GREEN}Новая модель лучше — сохраняем.{RESET}")
                else:
                    logger.info(f"{RED}Старая модель лучше — не сохраняем новую.{RESET}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить старую модель: {e}. Сохраняем новую.")
                save_model, is_best = True, True
        else:
            is_best = True

        if save_model:
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            joblib.dump(
                {
                    "model": final_model,
                    "threshold": best_threshold,
                    "metrics": final_metrics,
                    "selected_features": selected_features,
                },
                model_output_path,
            )
            logger.info(f"Модель сохранена в {model_output_path}")

        # логирование в MLflow
        log_with_mlflow(
            final_model=final_model,
            metric=metric,
            model_params=model_params,
            best_threshold=best_threshold,
            study=study,
            X_test=X_test_sel,
            y_test=y_test,
            final_metrics=final_metrics,
            final_metrics_train=final_metrics_train,
            selected_features=selected_features,
            model_output_path=model_output_path,
            run_name=run_name,
            n_trials=n_trials,
            input_example=input_example,
            y_pred_proba=y_test_proba,
            y_pred=(y_test_proba >= best_threshold).astype(int),
            is_best=is_best,
            fn_weight=fn_weight,
            fp_weight=fp_weight,
        )

    except Exception as e:
        logger.exception(f"Ошибка в run_optuna_experiment: {e}")
        raise


def log_with_mlflow(
    final_model,
    metric,
    model_params,
    best_threshold,
    study,
    X_test,
    y_test,
    final_metrics,
    final_metrics_train,
    selected_features,
    model_output_path,
    run_name,
    n_trials,
    input_example,
    y_pred_proba,
    y_pred,
    is_best: bool = False,
    fn_weight: int = 10,
    fp_weight: int = 2,
):
    try:
        n_selected_features = len(selected_features)
        cm_test = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm_test)

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            # Логируем параметры и метрики
            mlflow.log_params(model_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("model_type", "LGBMClassifier")
            mlflow.log_param("threshold", round(best_threshold, 4))
            mlflow.log_param("is_best", is_best)
            mlflow.log_param("opt_metric", f"{metric}")
            mlflow.log_param("n_selected_features", n_selected_features)
            mlflow.log_param("fn_weight", fn_weight)
            mlflow.log_param("fp_weight", fp_weight)

            mlflow.log_metric("fn_test", fn)
            mlflow.log_metric("fp_test", fp)
            mlflow.log_metric("f1_train", round(final_metrics_train["f1"], 3))
            mlflow.log_metric("f1_test", round(final_metrics["f1"], 3))
            mlflow.log_metric("f2_train", round(final_metrics_train["f2"], 3))
            mlflow.log_metric("f2_test", round(final_metrics["f2"], 3))
            mlflow.log_metric("accuracy_train", round(final_metrics_train["accuracy"], 3))
            mlflow.log_metric("accuracy_test", round(final_metrics["accuracy"], 3))
            mlflow.log_metric("recall_train", round(final_metrics_train["recall"], 3))
            mlflow.log_metric("recall_test", round(final_metrics["recall"], 3))
            mlflow.log_metric("precision_train", round(final_metrics_train["precision"], 3))
            mlflow.log_metric("precision_test", round(final_metrics["precision"], 3))
            mlflow.log_metric("roc_auc_train", round(final_metrics_train["roc_auc"], 3))
            mlflow.log_metric("roc_auc_test", round(final_metrics["roc_auc"], 3))
            mlflow.log_metric("cv_custom_train", round(study.best_value, 4))
            mlflow.log_metric("custom_train", round(final_metrics_train["custom"], 4))
            mlflow.log_metric("custom_test", round(final_metrics["custom"], 4))

            # Логируем модель как артефакт и через sklearn
            mlflow.log_artifact(model_output_path)
            mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)  # type: ignore

            # Confusion matrix
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
            fig_cm.tight_layout()
            fig_cm.savefig("confusion_matrix.png")
            plt.close(fig_cm)
            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")

            # ROC curve
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc)
            fig_roc.tight_layout()
            fig_roc.savefig("roc_curve.png")
            plt.close(fig_roc)
            mlflow.log_artifact("roc_curve.png")
            os.remove("roc_curve.png")

            # SHAP с fallback
            try:
                if hasattr(final_model, "feature_names_in_"):
                    model_feats = list(final_model.feature_names_in_)
                    X_test = X_test.reindex(columns=model_feats, fill_value=0)

                # Для LightGBM используем booster_
                booster = getattr(final_model, "booster_", final_model)
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer(X_test)

            except (json.JSONDecodeError, Exception) as e:
                print(f"Ошибка TreeExplainer: {e}. Переключаемся на универсальный Explainer.")
                explainer = shap.Explainer(final_model.predict, X_test)
                shap_values = explainer(X_test)

            # Унификация формата shap_values
            if hasattr(shap_values, "values"):  # старый API
                shap_array = shap_values.values
            else:  # новый API
                shap_array = shap_values

            # Для бинарной классификации — берём класс 1
            if shap_array.ndim == 3 and shap_array.shape[2] == 2:
                shap_class_1 = shap_array[:, :, 1]
            else:
                shap_class_1 = shap_array

            # Рисуем dot plot
            plt.figure()
            shap.summary_plot(shap_class_1, X_test, plot_type="dot", show=False, max_display=39)
            plt.tight_layout()
            plt.savefig("shap_dot_plot.png")
            plt.close()

            # Логгируем артефакт в MLflow
            mlflow.log_artifact("shap_dot_plot.png")
            os.remove("shap_dot_plot.png")

            # Threshold vs metrics
            f1s, precisions, recalls = [], [], []
            for t in THRESHOLDS:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred_temp, average="binary", zero_division=0
                )
                f1s.append(f1)
                precisions.append(p)
                recalls.append(r)

            plt.figure(figsize=(8, 6))
            plt.plot(THRESHOLDS, f1s, label="F1")
            plt.plot(THRESHOLDS, precisions, label="Precision")
            plt.plot(THRESHOLDS, recalls, label="Recall")
            plt.axvline(
                float(best_threshold),
                color="gray",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Threshold vs F1 / Precision / Recall")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("threshold_metrics.png")
            plt.close()
            mlflow.log_artifact("threshold_metrics.png")
            os.remove("threshold_metrics.png")

            # Histogram of predicted probabilities
            plt.figure(figsize=(8, 6))
            plt.hist(y_pred_proba, bins=50, alpha=0.7)
            plt.axvline(
                float(best_threshold),
                color="red",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.title("Distribution of predicted probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("proba_distribution.png")
            plt.close()
            mlflow.log_artifact("proba_distribution.png")
            os.remove("proba_distribution.png")

            # TP/FP/FN/TN vs Threshold plot
            tps, fps, fns, tns = [], [], [], []

            for t in THRESHOLDS:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                cm_temp = confusion_matrix(y_test, y_pred_temp, labels=[0, 1])
                tn, fp, fn, tp = get_confusion_counts(cm_temp)
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
                tns.append(tn)

            plt.figure(figsize=(8, 6))
            plt.plot(THRESHOLDS, tps, label="TP")
            plt.plot(THRESHOLDS, fps, label="FP")
            plt.plot(THRESHOLDS, fns, label="FN")
            plt.plot(THRESHOLDS, tns, label="TN")
            plt.axvline(
                float(best_threshold),
                color="gray",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.xlabel("Threshold")
            plt.ylabel("Count")
            plt.title("TP / FP / FN / TN vs Threshold")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("threshold_confusion_counts.png")
            plt.close()
            mlflow.log_artifact("threshold_confusion_counts.png")
            os.remove("threshold_confusion_counts.png")

            selected_features_path = "selected_features.txt"
            with open(selected_features_path, "w") as f:
                for feat in selected_features:
                    f.write(f"{feat}\n")

            mlflow.log_param("n_selected_features", n_selected_features)
            mlflow.log_artifact(selected_features_path)
            os.remove(selected_features_path)

            # Correlation heatmap
            safe_features = [f for f in selected_features if f in X_test.columns]
            if safe_features:
                corr_matrix = X_test[safe_features].corr(method="pearson")
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    cbar_kws={"shrink": 0.75},
                    linewidths=0.5,
                    linecolor="gray",
                    annot_kws={"size": 6},
                )
                plt.title("Correlation Heatmap (Test Data)")
                plt.tight_layout()
                plt.savefig("correlation_heatmap.png")
                plt.close()
                mlflow.log_artifact("correlation_heatmap.png")
                os.remove("correlation_heatmap.png")

                # Log high correlation feature pairs (|corr| > 0.9)
                high_corr_output = "high_corr_pairs.txt"
                corr_abs = corr_matrix.abs()
                upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

                with open(high_corr_output, "w") as f:
                    for col in upper.columns:
                        for row in upper.index:
                            val = upper.loc[row, col]
                            if pd.notnull(val) and val > 0.9:
                                f.write(f"{row} - {col}: {val:.3f}\n")

                mlflow.log_artifact(high_corr_output)
                os.remove(high_corr_output)

            params_path = "best_params.json"
            with open(params_path, "w") as f:
                json.dump(model_params, f, indent=4)

            mlflow.log_artifact(params_path)
            os.remove(params_path)

            experiment_config = {
                "TEST_SIZE": TEST_SIZE,
                "RANDOM_STATE": RANDOM_STATE,
                "N_TRIALS": N_TRIALS,
                "N_SPLITS": N_SPLITS,
                "METRIC": METRIC,
                "TARGET_COL": TARGET_COL,
                "FN_WEIGHT": fn_weight,
                "FP_WEIGHT": fp_weight,
                "MIN_PRECISION": MIN_PRECISION,
            }

            with open("experiment_config.json", "w") as f:
                json.dump(experiment_config, f, indent=4)
            mlflow.log_artifact("experiment_config.json")
            os.remove("experiment_config.json")

            # Log Optuna optimization progress (trial score per iteration)
            scores = [trial.value for trial in study.trials if trial.value is not None]

            plt.figure(figsize=(10, 6))
            plt.plot(scores, marker="o", linestyle="-", alpha=0.8)
            plt.xlabel("Trial Number")
            plt.ylabel("Score")
            plt.title("Optuna Optimization Progress")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("optuna_progress.png")
            plt.close()
            mlflow.log_artifact("optuna_progress.png")
            os.remove("optuna_progress.png")

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def keep_top_n_models(experiment_name: str, metric: str, n_keep: int = 10):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    # Получаем все запуски
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],  # сортировка по убыванию метрики
    )

    # Если моделей больше n_keep → удаляем лишние
    if len(runs) > n_keep:
        for run in runs[n_keep:]:
            client.delete_run(run.info.run_id)


def today():
    return datetime.now()


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/root6/python/dismissal_predict_v2/mlruns")

    # Преобразуем данные один раз
    main_users = convert_all_to_float(main_users, exclude_cols=[TARGET_COL])
    top_users = convert_all_to_float(top_users, exclude_cols=[TARGET_COL])
    main_users[TARGET_COL] = (
        pd.to_numeric(main_users[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )
    top_users[TARGET_COL] = (
        pd.to_numeric(top_users[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )
    main_users = main_users.drop_duplicates()
    top_users = top_users.drop_duplicates()

    X_main = main_users.drop(columns=[TARGET_COL])
    y_main = main_users[TARGET_COL]
    X_top = top_users.drop(columns=[TARGET_COL])
    y_top = top_users[TARGET_COL]

    n_iter = 0
    for fn_weight, fp_weight in product(FN_WEIGHT_GRID, FP_WEIGHT_GRID):
        n_iter += 1
        logger.info(f"=== Запуск с параметрами: FN_WEIGHT={fn_weight}, FP_WEIGHT={fp_weight} ===")

        # train/test split main_users
        X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
            X_main, y_main, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_main
        )

        run_optuna_experiment(
            X_train=X_train_main,
            y_train=y_train_main,
            X_test=X_test_main,
            y_test=y_test_main,
            metric=METRIC,
            n_trials=N_TRIALS,
            experiment_name=f"{MLFLOW_EXPERIMENT_MAIN}",
            model_output_path=f"{MODELS}/main_users.pkl",
            current_time=today(),
            fn_weight=fn_weight,
            fp_weight=fp_weight,
            n_iter=n_iter,
        )

        # train/test split top_users
        X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
            X_top, y_top, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_top
        )

        # run_optuna_experiment(
        #     X_train=X_train_top,
        #     y_train=y_train_top,
        #     X_test=X_test_top,
        #     y_test=y_test_top,
        #     metric=METRIC,
        #     n_trials=N_TRIALS,
        #     experiment_name=f"{MLFLOW_EXPERIMENT_TOP}",
        #     model_output_path=f"{MODELS}/top_users.pkl",
        #     current_time=today(),
        #     fn_weight=fn_weight,
        #     fp_weight=fp_weight,
        # )

    keep_top_n_models(
        experiment_name=f"{MLFLOW_EXPERIMENT_MAIN}", metric=METRIC, n_keep=N_MLFLOW_COUNT
    )
    # keep_top_n_models(experiment_name=f"{MLFLOW_EXPERIMENT_TOP}", metric=METRIC, n_keep=N_MLFLOW_COUNT)
