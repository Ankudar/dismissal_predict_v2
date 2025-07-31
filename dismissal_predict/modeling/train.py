from collections import Counter
from datetime import datetime
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
import shap  # type: ignore
from sklearn.exceptions import UndefinedMetricWarning
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
from xgboost import XGBClassifier

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
N_TRIALS = 2  # итерации для оптуны
N_SPLITS = 5  # число кроссвалидаций
METRIC = "custom"
EVAL_METRIC = "logloss"
MLFLOW_EXPERIMENT_MAIN = "xgboost_main_users"
MLFLOW_EXPERIMENT_TOP = "xgboost_top_users"

TARGET_COL = "уволен"


warnings.filterwarnings("ignore")

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")


def custom_metric_from_counts(tp, tn, fn, fp):
    total = tp + tn + fn + fp
    if total == 0:
        return 0
    score = (tp + tn) - 1.5 * fn - 0.5 * fp
    return score / total


def is_new_model_better(new_metrics, old_metrics, delta=0.001):
    def round3(x):
        return round(x or 0, 3)

    new_f1 = round3(new_metrics.get("f1"))
    old_f1 = round3(old_metrics.get("f1"))

    if new_f1 > old_f1:
        return True
    if new_f1 < old_f1:
        return False

    # f1 равны — сравниваем recall
    new_recall = round3(new_metrics.get("recall"))
    old_recall = round3(old_metrics.get("recall"))

    if new_recall > old_recall:
        return True
    if new_recall < old_recall:
        return False

    # recall равны — сравниваем precision
    new_precision = round3(new_metrics.get("precision"))
    old_precision = round3(old_metrics.get("precision"))

    return new_precision > old_precision


def manual_optuna_progress(study, n_trials, func):
    for _ in tqdm(range(n_trials), desc="Optuna Tuning"):
        study.optimize(func, n_trials=1, catch=(Exception,), n_jobs=4)


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


def cross_val_best_threshold(
    model,
    X,
    y,
    thresholds=np.arange(0.01, 0.9, 0.01),
    metric=METRIC,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    print(f"Поиск threshold по метрике {metric.upper()} с минимальными FN + FP...\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_val)[:, 1]

        best_t = None
        best_metric = -1
        best_fn_fp = float("inf")
        best_fn = best_fp = None

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            fn_fp = fn + fp

            if metric == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == "roc_auc":
                score = roc_auc_score(y_val, y_probs)
            elif metric == "custom":
                score = custom_metric_from_counts(tp, tn, fn, fp)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if score > best_metric or (score == best_metric and fn_fp < best_fn_fp):
                best_metric = score
                best_t = t
                best_fn_fp = fn_fp
                best_fn = fn
                best_fp = fp

        fold_results.append(
            {
                "fold": fold,
                "threshold": best_t,
                "metric": best_metric,
                "fn": best_fn,
                "fp": best_fp,
                "fn_fp": best_fn_fp,
            }
        )

        print(
            f"[Fold {fold}] ✅ threshold = {best_t:.3f} → {metric} = {best_metric:.4f}, FN+FP = {best_fn_fp}"
        )

    # Сортировка по метрике по убыванию, далее по fn+fp по возрастанию
    fold_results = sorted(fold_results, key=lambda x: (-x["metric"], x["fn_fp"]))

    top_k = n_splits // 3  # часть фолдов
    top_folds = fold_results[:top_k]

    mean_t = np.mean([r["threshold"] for r in top_folds])
    mean_metric = np.mean([r["metric"] for r in top_folds])
    mean_fn = np.mean([r["fn"] for r in top_folds])
    mean_fp = np.mean([r["fp"] for r in top_folds])

    print(
        f"\nФинальный threshold = {mean_t:.3f} → {metric} = {mean_metric:.4f}, средние FN = {mean_fn:.1f}, FP = {mean_fp:.1f} (по top-{top_k} фолдам)"
    )
    return mean_t, mean_metric, mean_fn, mean_fp


def custom_cv_score(model, X, y, threshold, n_splits=N_SPLITS, metric=METRIC):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        y_probs = model.predict_proba(X_valid_fold)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)

                if metric == "f1":
                    score = f1_score(y_valid_fold, y_preds)
                elif metric == "recall":
                    score = recall_score(y_valid_fold, y_preds)
                elif metric == "precision":
                    score = precision_score(y_valid_fold, y_preds)
                elif metric == "roc_auc":
                    score = roc_auc_score(y_valid_fold, y_probs)
                elif metric == "custom":
                    cm = confusion_matrix(y_valid_fold, y_preds, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
                    score = custom_metric_from_counts(tp, tn, fn, fp)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            if not np.isnan(score):
                scores.append(score)
        except Exception as e:
            print(f"Ошибка на фолде: {e}")
            continue

    return np.mean(scores) if scores else float("nan")


def split_df(data):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop([TARGET_COL], axis=1),
            data[TARGET_COL],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=data[TARGET_COL],
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def objective(trial, X_train, y_train, threshold):
    try:
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "eval_metric": EVAL_METRIC,
        }

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        recalls, precisions, f1s, accuracies = [], [], [], []
        fn_list, fp_list, tn_list, tp_list = [], [], [], []

        for train_idx, valid_idx in skf.split(X_train, y_train):
            model = XGBClassifier(**params)
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            recalls.append(recall_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_val, y_pred))

            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tp = tn = fp = fn = 0
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

            fn_list.append(fn)
            fp_list.append(fp)
            tn_list.append(tn)
            tp_list.append(tp)

        # Средние значения по фолдам
        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        mean_f1 = np.mean(f1s)
        mean_accuracy = np.mean(accuracies)

        # Выбор метрики
        if METRIC == "f1":
            score = mean_f1
        elif METRIC == "accuracy":
            score = mean_accuracy
        elif METRIC == "recall":
            score = mean_recall
        elif METRIC == "precision":
            score = mean_precision
        elif METRIC == "custom":
            score = custom_metric_from_counts(
                tp=int(np.mean(tp_list)),
                tn=int(np.mean(tn_list)),
                fn=int(np.mean(fn_list)),
                fp=int(np.mean(fp_list)),
            )
        else:
            logger.warning(f"Неизвестная метрика '{METRIC}', используется recall по умолчанию.")
            score = mean_recall

        logger.info(
            f"Trial {trial.number} → "
            f"Recall: {mean_recall:.3f}, Precision: {mean_precision:.3f}, F1: {mean_f1:.3f}, "
            f"Accuracy: {mean_accuracy:.3f}, Score: {score:.3f} | "
            f"FN: {np.mean(fn_list):.1f}, FP: {np.mean(fp_list):.1f}, TN: {np.mean(tn_list):.1f}, TP: {np.mean(tp_list):.1f}"
        )

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
    global_threshold,
):
    try:
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        def optuna_objective(trial):
            return objective(trial, X_train, y_train, global_threshold)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        manual_optuna_progress(study, n_trials, optuna_objective)

        best_params = study.best_trial.params

        final_model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric=EVAL_METRIC,
        )
        final_model.fit(X_train, y_train)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= global_threshold).astype(int)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tp = tn = fp = fn = 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

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
                logger.info(f"Старая модель: {old_metrics}")
                logger.info(f"Новая модель: {final_metrics}")
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
                    "threshold": global_threshold,
                    "metrics": final_metrics,
                },
                model_output_path,
            )
            logger.info(f"Модель сохранена в {model_output_path}")

        input_example = pd.DataFrame(X_test[:1], columns=X_test.columns)
        run_name = f"model_{current_time}"

        # Метрики на трейне
        y_pred_proba_train = final_model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_pred_proba_train >= global_threshold).astype(int)
        recall_train = recall_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
        tp = tn = fp = fn = 0
        if cm_train.shape == (2, 2):
            tn, fp, fn, tp = cm_train.ravel()

        final_metrics_train = {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "precision": precision_train,
            "recall": recall_train,
            "roc_auc": roc_auc_score(y_train, y_pred_proba_train),
            "f1": f1_score(y_train, y_pred_train),
            "custom": custom_metric_from_counts(tp, tn, fn, fp),
        }

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("model_type", "XGBoostClassifier")
            mlflow.log_param("threshold", round(global_threshold, 4))

            mlflow.log_param("opt_metric", f"{METRIC}")

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

            mlflow.log_metric("custom_train", round(final_metrics_train["custom"], 3))
            mlflow.log_metric("custom_test", round(final_metrics["custom"], 3))

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

            # SHAP summary
            explainer = shap.Explainer(final_model)
            shap_values = explainer(X_test)
            plt.figure()  # важно!
            shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
            plt.tight_layout()
            plt.savefig("shap_dot_plot.png")
            plt.close()
            mlflow.log_artifact("shap_dot_plot.png")
            os.remove("shap_dot_plot.png")

            # Threshold vs metrics
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1s, precisions, recalls = [], [], []
            for t in thresholds:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred_temp, average="binary", zero_division=0
                )
                f1s.append(f1)
                precisions.append(p)
                recalls.append(r)

            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, f1s, label="F1")
            plt.plot(thresholds, precisions, label="Precision")
            plt.plot(thresholds, recalls, label="Recall")
            plt.axvline(
                global_threshold,
                color="gray",
                linestyle="--",
                label=f"Threshold = {round(global_threshold, 3)}",
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
                global_threshold,
                color="red",
                linestyle="--",
                label=f"Threshold = {round(global_threshold, 3)}",
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

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def today():
    now = pd.to_datetime(datetime.now())
    return now


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/root6/python/dismissal_predict_v2/mlruns")

    # Преобразуем данные
    main_users = convert_all_to_float(main_users, exclude_cols=[TARGET_COL])
    top_users = convert_all_to_float(top_users, exclude_cols=[TARGET_COL])
    main_users[TARGET_COL] = (
        pd.to_numeric(main_users[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )
    top_users[TARGET_COL] = (
        pd.to_numeric(top_users[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )

    # Удаляем дубликаты
    main_users = main_users.drop_duplicates()
    top_users = top_users.drop_duplicates()

    # Сплитим данные один раз!
    X_main = main_users.drop(columns=[TARGET_COL])
    y_main = main_users[TARGET_COL]
    X_top = top_users.drop(columns=[TARGET_COL])
    y_top = top_users[TARGET_COL]

    # --- MAIN USERS ---
    base_model_main = XGBClassifier(
        scale_pos_weight=Counter(y_main)[0] / Counter(y_main)[1],
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
    )

    global_threshold_main, f1, fn, fp = cross_val_best_threshold(
        base_model_main, X_main, y_main, metric=METRIC
    )

    print(
        f"Выбранный threshold для all = {global_threshold_main:.3f}, f1 = {f1:.4f}, fn = {fn:.4f}, fp = {fp:.4f}"
    )

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
        experiment_name=MLFLOW_EXPERIMENT_MAIN,
        model_output_path=f"{MODELS}/xgb_main_users.pkl",
        current_time=today(),
        global_threshold=global_threshold_main,
    )

    # --- TOP USERS ---
    base_model_top = XGBClassifier(
        scale_pos_weight=Counter(y_top)[0] / Counter(y_top)[1],
        random_state=RANDOM_STATE,
        eval_metric=EVAL_METRIC,
    )

    global_threshold_top, f1, fn, fp = cross_val_best_threshold(
        base_model_top, X_top, y_top, metric=METRIC
    )

    print(
        f"Выбранный threshold для top = {global_threshold_top:.3f}, f1 = {f1:.4f}, fn = {fn:.4f}, fp = {fp:.4f}"
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
        n_trials=N_TRIALS,
        experiment_name=MLFLOW_EXPERIMENT_TOP,
        model_output_path=f"{MODELS}/xgb_top_users.pkl",
        current_time=today(),
        global_threshold=global_threshold_top,
    )
