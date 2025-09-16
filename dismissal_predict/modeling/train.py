from datetime import datetime
from itertools import product
import json
import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
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

TEST_SIZE = 0.25
RANDOM_STATE = 40
N_TRIALS = 5000  # итерации для оптуны
N_TRIALS_FOR_TOP = 5000
N_SPLITS = 5  # число кроссвалидаций
METRIC = "custom"
MLFLOW_EXPERIMENT_MAIN = "main_users"
MLFLOW_EXPERIMENT_TOP = "top_users"
EARLY_STOP = 100

TARGET_COL = "уволен"

N_JOBS = 1
THRESHOLDS = np.arange(0.1, 0.9, 0.01)

warnings.filterwarnings("ignore")

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


def custom_metric_from_counts(tp, tn, fn, fp):
    fn_penalty = np.exp(-FN_PENALTY_WEIGHT * fn)
    fp_penalty = np.exp(-FP_PENALTY_WEIGHT * fp)
    score = fn_penalty * FN_WEIGHT + fp_penalty * FP_WEIGHT
    max_score = FN_WEIGHT + FP_WEIGHT
    normalized_score = score / max_score
    return round(normalized_score, 6)


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


def objective(trial, X_train, y_train, all_columns):
    try:
        k_best = trial.suggest_int("k_best", 5, min(40, X_train.shape[1]))

        # Сохраняем имена фич до преобразования
        feature_names = X_train.columns.tolist()

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        recalls, precisions, f1s, accuracies, roc_auc = [], [], [], [], []
        fn_list, fp_list, tn_list, tp_list = [], [], [], []
        fold_thresholds = []
        all_selected_features = []  # Для сбора фич со всех фолдов

        for train_idx, valid_idx in skf.split(X_train, y_train):
            # Разделяем данные ДО feature selection
            X_tr_raw, X_val_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            # Feature selection на ТРЕНИРОВОЧНОМ фолде (без утечки!)
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_tr = selector.fit_transform(X_tr_raw, y_tr)
            X_val = selector.transform(X_val_raw)

            # Получаем выбранные фичи для этого фолда
            selected_idx = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_idx]
            all_selected_features.append(selected_features)

            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]

            best_fp = float("inf")
            best_fn = float("inf")
            best_score = -np.inf
            best_threshold = 0.5

            for t in THRESHOLDS:
                y_pred_t = (y_proba >= t).astype(int)
                cm_t = confusion_matrix(y_val, y_pred_t, labels=[0, 1])
                tn, fp, fn, tp = get_confusion_counts(cm_t)

                if fn > FN_STOP:
                    continue

                precision_t = precision_score(y_val, y_pred_t, zero_division=0)
                if precision_t < MIN_PRECISION:
                    continue

                score_t = None
                if METRIC == "f1":
                    score_t = f1_score(y_val, y_pred_t, zero_division=0)
                elif METRIC == "accuracy":
                    score_t = accuracy_score(y_val, y_pred_t)
                elif METRIC == "recall":
                    score_t = recall_score(y_val, y_pred_t)
                elif METRIC == "precision":
                    score_t = precision_score(y_val, y_pred_t, zero_division=0)
                elif METRIC == "roc_auc":
                    score_t = roc_auc_score(y_val, y_proba)
                elif METRIC == "custom":
                    score_t = custom_metric_from_counts(tp, tn, fn, fp)
                else:
                    logger.warning(
                        f"Неизвестная метрика '{METRIC}', используем recall по умолчанию"
                    )
                    score_t = recall_score(y_val, y_pred_t)

                if score_t is None:
                    continue

                if fn <= best_fn + MAX_FN_SOFT and (
                    fn < best_fn
                    or (fn == best_fn and fp < best_fp)
                    or (fn == best_fn and fp == best_fp and score_t > best_score)
                ):
                    best_fp = fp
                    best_fn = fn
                    best_score = score_t
                    best_threshold = t

            fold_thresholds.append(best_threshold)
            y_pred = (y_proba >= best_threshold).astype(int)

            recalls.append(recall_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_val, y_pred))
            roc_auc.append(roc_auc_score(y_val, y_proba))

            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = get_confusion_counts(cm)

            fn_list.append(fn)
            fp_list.append(fp)
            tn_list.append(tn)
            tp_list.append(tp)

        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        mean_f1 = np.mean(f1s)
        mean_accuracy = np.mean(accuracies)
        mean_roc_auc = np.mean(roc_auc)

        # Выбираем наиболее частые фичи по всем фолдам
        from collections import Counter

        flat_features = [f for sublist in all_selected_features for f in sublist]
        feature_counts = Counter(flat_features)
        most_common_features = [feature for feature, count in feature_counts.most_common(k_best)]

        trial.set_user_attr("selected_features", most_common_features)
        trial.set_user_attr("n_selected_features", len(most_common_features))

        if METRIC == "f1":
            score = mean_f1
        elif METRIC == "accuracy":
            score = mean_accuracy
        elif METRIC == "recall":
            score = mean_recall
        elif METRIC == "precision":
            score = mean_precision
        elif METRIC == "roc_auc":
            score = mean_roc_auc
        elif METRIC == "custom":
            # Суммируем по всем фолдам, а не усредняем!
            tp_total = np.sum(tp_list)
            tn_total = np.sum(tn_list)
            fn_total = np.sum(fn_list)
            fp_total = np.sum(fp_list)
            score = custom_metric_from_counts(tp_total, tn_total, fn_total, fp_total)
        else:
            logger.warning(f"Неизвестная метрика '{METRIC}', используется recall по умолчанию.")
            score = mean_recall

        logger.info(
            f"\nTrial {trial.number} → k_best: {k_best}\n"
            f"Recall: {mean_recall:.3f}, Precision: {mean_precision:.3f}, F1: {mean_f1:.3f},\n"
            f"Accuracy: {mean_accuracy:.3f}, Score: {score:.3f}\n"
            f"FN: {np.mean(fn_list):.1f}, FP: {np.mean(fp_list):.1f}, TN: {np.mean(tn_list):.1f}, TP: {np.mean(tp_list):.1f}\n"
        )

        final_threshold = np.mean(fold_thresholds)
        trial.set_user_attr("best_threshold", final_threshold)

        if np.isnan(score) or np.isinf(score):
            return -1
        return score

    except Exception as e:
        logger.exception(f"Ошибка в objective: {e}")
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
):
    try:
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        def optuna_objective(trial):
            return objective(trial, X_train, y_train, X_train.columns)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        early_stopping = EarlyStoppingCallback(
            patience=50, min_delta=0.001
        )  # Это и строку ниже закоментить чтобы убрать раннюю остановку
        study.optimize(optuna_objective, n_trials=n_trials, callbacks=[early_stopping])

        # manual_optuna_progress(study, n_trials, optuna_objective) # включить если выключили раннюю остановка

        best_params = study.best_trial.params.copy()

        selected_features = study.best_trial.user_attrs["selected_features"]
        n_selected_features = study.best_trial.user_attrs["n_selected_features"]
        best_params.pop("k_best", None)

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        best_threshold = study.best_trial.user_attrs["best_threshold"]

        final_model = RandomForestClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

        final_model.fit(X_train, y_train)

        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm)

        final_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
            "custom": custom_metric_from_counts(tp, tn, fn, fp),
        }

        save_model = True
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
                if save_model:
                    logger.info(f"{GREEN}Новая модель лучше — сохраняем.{RESET}")
                else:
                    logger.info(f"{RED}Старая модель лучше — не сохраняем новую.{RESET}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить старую модель: {e}. Сохраняем новую.")
                save_model = True

        if save_model:
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

        input_example = pd.DataFrame(X_test[:1], columns=X_test.columns)
        run_name = f"model_{current_time}"

        y_pred_proba_train = final_model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_pred_proba_train >= best_threshold).astype(int)
        recall_train = recall_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
        tp = tn = fp = fn = 0
        tn, fp, fn, tp = get_confusion_counts(cm_train)

        final_metrics_train = {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "precision": precision_train,
            "recall": recall_train,
            "roc_auc": roc_auc_score(y_train, y_pred_proba_train),
            "f1": f1_score(y_train, y_pred_train),
            "custom": custom_metric_from_counts(tp, tn, fn, fp),
        }

        log_with_mlflow(
            final_model=final_model,
            metric=metric,
            best_params=best_params,
            best_threshold=best_threshold,
            study=study,
            X_test=X_test,
            y_test=y_test,
            final_metrics=final_metrics,
            final_metrics_train=final_metrics_train,
            selected_features=selected_features,
            model_output_path=model_output_path,
            run_name=f"model_{current_time}",
            n_trials=n_trials,
            input_example=input_example,
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
        )

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def log_with_mlflow(
    final_model,
    metric,
    best_params,
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
):
    try:
        n_selected_features = len(selected_features)
        cm_test = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm_test)

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("threshold", round(best_threshold, 4))
            mlflow.log_param(
                "cv_best_threshold", round(study.best_trial.user_attrs["best_threshold"], 4)
            )

            mlflow.log_param("opt_metric", f"{metric}")
            mlflow.log_metric("fn_test", fn)
            mlflow.log_metric("fp_test", fp)

            mlflow.log_metric("f1_train", round(final_metrics_train["f1"], 3))
            mlflow.log_metric("f1_test", round(final_metrics["f1"], 3))

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

            # SHAP
            if hasattr(final_model, "feature_names_in_"):
                X_test = X_test[final_model.feature_names_in_]

            explainer = shap.Explainer(final_model, X_test)
            shap_values = explainer(X_test, check_additivity=False)

            # Берем SHAP только для класса 1
            shap_class_1 = shap_values.values[:, :, 1]

            # Строим dot plot
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
            corr_matrix = X_test[selected_features].corr(method="pearson")
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
                annot_kws={"size": 6},  # <-- Уменьшенный шрифт аннотаций
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
                json.dump(best_params, f, indent=4)

            mlflow.log_artifact(params_path)
            os.remove(params_path)

            experiment_config = {
                "TEST_SIZE": TEST_SIZE,
                "RANDOM_STATE": RANDOM_STATE,
                "N_TRIALS": N_TRIALS,
                "N_TRIALS_FOR_TOP": N_TRIALS_FOR_TOP,
                "N_SPLITS": N_SPLITS,
                "METRIC": METRIC,
                "TARGET_COL": TARGET_COL,
                "FN_PENALTY_WEIGHT": FN_PENALTY_WEIGHT,
                "FP_PENALTY_WEIGHT": FP_PENALTY_WEIGHT,
                "FN_WEIGHT": FN_WEIGHT,
                "FP_WEIGHT": FP_WEIGHT,
                "MIN_PRECISION": MIN_PRECISION,
                "FN_STOP": FN_STOP,
                "MAX_FN_SOFT": MAX_FN_SOFT,
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

    # Сетка параметров
    fn_penalty_grid = range(4, 6)  # первое входит, второе нет
    fp_penalty_grid = range(2, 4)
    fn_stop_grid = range(0, 2)
    max_fn_soft_grid = range(0, 2)

    # FN_PENALTY_WEIGHT: Увеличение этого значения делает штраф за ложные отрицательные более значительным, что помогает минимизировать их количество.
    # FP_PENALTY_WEIGHT: Уменьшение этого значения снижает штраф за ложные положительные, что позволяет им быть менее критичными.
    # FN_WEIGHT и FP_WEIGHT: Увеличение веса для FN и уменьшение для FP помогает сбалансировать итоговый результат.
    # FN_STOP # Жёсткое ограничение FN для подбора трешхолда
    # MAX_FN_SOFT # Мягкое ограничение FN уже непосредственно в модели обучения

    for fn_penalty, fp_penalty, fn_stop_val, max_fn_soft_val in product(
        fn_penalty_grid, fp_penalty_grid, fn_stop_grid, max_fn_soft_grid
    ):
        # Устанавливаем глобальные параметры
        FN_PENALTY_WEIGHT = fn_penalty
        FP_PENALTY_WEIGHT = fp_penalty
        FN_WEIGHT = 0.7
        FP_WEIGHT = 0.3
        MIN_PRECISION = 0.3
        FN_STOP = fn_stop_val
        MAX_FN_SOFT = max_fn_soft_val

        logger.info(
            f"=== Запуск с параметрами: FN_PENALTY_WEIGHT={FN_PENALTY_WEIGHT}, "
            f"FP_PENALTY_WEIGHT={FP_PENALTY_WEIGHT}, FN_STOP={FN_STOP}, MAX_FN_SOFT={MAX_FN_SOFT} ==="
        )

        # train/test split
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
        )

        X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
            X_top, y_top, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_top
        )

        run_optuna_experiment(
            X_train=X_train_top,
            y_train=y_train_top,
            X_test=X_test_top,
            y_test=y_test_top,
            metric=METRIC,
            n_trials=N_TRIALS_FOR_TOP,
            experiment_name=f"{MLFLOW_EXPERIMENT_TOP}",
            model_output_path=f"{MODELS}/top_users.pkl",
            current_time=today(),
        )
