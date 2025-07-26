from collections import Counter
from datetime import datetime
from itertools import product
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
MODELS = "/home/root6/python/dismissal_predict_v2/models"
os.makedirs(MODELS, exist_ok=True)

INPUT_FILE_MAIN_USERS = f"{DATA_PROCESSED}/main_users_for_train.csv"
INPUT_FILE_TOP_USERS = f"{DATA_PROCESSED}/main_top_for_train.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 40
N_TRIALS = 1000
MLFLOW_EXPERIMENT_MAIN = "xgboost_main_users"
MLFLOW_EXPERIMENT_TOP = "xgboost_top_users"

TARGET_COL = "уволен"

COST_FP_NUM = 1
COST_FN_NUM = 20
N_SPLITS = 5


warnings.filterwarnings("ignore")

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")


def experiment_cost_matrix(X, y):
    cost_fp_values = list(range(1, 21, 2))
    cost_fn_values = list(range(20, 201, 20))

    results = []

    for cost_fp, cost_fn in product(cost_fp_values, cost_fn_values):
        logger.info(f"Пробуем cost_fp={cost_fp}, cost_fn={cost_fn}")

        model = XGBClassifier(
            scale_pos_weight=Counter(y)[0] / Counter(y)[1],
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        threshold = cross_val_best_threshold(model, X, y, cost_fp, cost_fn)
        score = custom_cv_score(model, X, y, threshold, cost_fp, cost_fn)

        results.append(
            {
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "threshold": round(threshold, 4),
                "custom_score": round(score, 3),
            }
        )

    df = pd.DataFrame(results).sort_values("custom_score", ascending=False)
    print("\n🔍 Результаты перебора cost_fp и cost_fn:")
    print(df.to_string(index=False))
    return df


def is_new_model_better(new_metrics, old_metrics, metric):
    return new_metrics[metric] > old_metrics.get(metric, 0)


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


def find_best_threshold(y_true, y_probs, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM):
    thresholds = np.linspace(0, 1, 1000)
    best_score = -np.inf
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()
        score = tp + tn - cost_fp * fp - cost_fn * fn

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold


def cross_val_best_threshold(
    model, X, y, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM, n_splits=N_SPLITS
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    thresholds = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        y_probs = model.predict_proba(X_valid_fold)[:, 1]
        best_thresh = find_best_threshold(y_valid_fold, y_probs, cost_fp, cost_fn)
        thresholds.append(best_thresh)

    return np.mean(thresholds)


def custom_cv_score(
    model, X, y, threshold, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM, n_splits=N_SPLITS
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        y_probs = model.predict_proba(X_valid_fold)[:, 1]

        y_preds = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y_valid_fold, y_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        score = tp + tn - cost_fp * fp - cost_fn * fn
        scores.append(score)

    return np.mean(scores)


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
            "n_estimators": trial.suggest_int("n_estimators", 10, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.7),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        score_mean = custom_cv_score(
            model, X_train, y_train, threshold=threshold, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM
        )
        logger.info(f"Trial {trial.number} finished with custom score: {score_mean:.4f}")
        return score_mean
    except Exception as e:
        logger.exception(f"Ошибка в objective: {e}")
        raise


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
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        global_threshold = cross_val_best_threshold(base_model, X_train, y_train)
        logger.info(f"Глобальный threshold: {global_threshold:.4f}")

        def optuna_objective(trial):
            return objective(trial, X_train, y_train, global_threshold)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        manual_optuna_progress(study, n_trials, optuna_objective)

        best_params = study.best_trial.params
        best_mean_score = study.best_value

        final_model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        final_model.fit(X_train, y_train)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]

        y_pred = (y_pred_proba >= global_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        custom_score = tp + tn - COST_FP_NUM * fp - COST_FN_NUM * fn

        final_metrics = {
            "custom_score": custom_score,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
        }

        save_model = True
        if os.path.exists(model_output_path):
            try:
                old_model_bundle = joblib.load(model_output_path)
                old_metrics = old_model_bundle.get("metrics", {})
                logger.info(f"Старая модель: {old_metrics}")
                logger.info(f"Новая модель: {final_metrics}")
                save_model = is_new_model_better(final_metrics, old_metrics, metric)

                if save_model:
                    logger.info("Новая модель лучше — сохраняем.")
                else:
                    logger.info("Старая модель лучше — не сохраняем новую.")
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
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("model_type", "XGBoostClassifier")
            mlflow.log_param("threshold", round(global_threshold, 4))
            mlflow.log_param("global_threshold", round(global_threshold, 4))
            mlflow.log_metric(f"train_{metric}", round(best_mean_score, 3))
            mlflow.log_metrics(
                {
                    **final_metrics,
                    "optimized_metric_value": round(float(final_metrics[metric]), 3),
                }
            )

            mlflow.log_artifact(model_output_path)
            mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)  # type: ignore

            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
            fig_cm.savefig("confusion_matrix.png")
            plt.close(fig_cm)
            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")

            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc)
            fig_roc.savefig("roc_curve.png")
            plt.close(fig_roc)
            mlflow.log_artifact("roc_curve.png")
            os.remove("roc_curve.png")

            explainer = shap.Explainer(final_model)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, plot_type="dot")
            plt.savefig("shap_dot_plot.png")
            plt.close()
            mlflow.log_artifact("shap_dot_plot.png")
            os.remove("shap_dot_plot.png")

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def today():
    now = pd.to_datetime(datetime.now())
    return now


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/root6/python/dismissal_predict_v2/mlruns")

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

    X_train_main, X_test_main, y_train_main, y_test_main = split_df(main_users)
    run_optuna_experiment(
        X_train_main,
        y_train_main,
        X_test_main,
        y_test_main,
        metric="custom_score",
        n_trials=N_TRIALS,
        experiment_name=MLFLOW_EXPERIMENT_MAIN,
        model_output_path=f"{MODELS}/xgb_main_users.pkl",
        current_time=today(),
    )

    # чтобы провести эксперименты по подбору штрафов - вначале надо закомментить run_optuna_experiment,
    # а после уже выставить лучшие показатели, расскомментить обучение и закомментить эксперимент
    # df_cost_matrix_results = experiment_cost_matrix(X_train_main, y_train_main)
    # оказалось все логично - минимальные значения = лучшие показатели

    X_train_top, X_test_top, y_train_top, y_test_top = split_df(top_users)
    run_optuna_experiment(
        X_train_top,
        y_train_top,
        X_test_top,
        y_test_top,
        metric="custom_score",
        n_trials=N_TRIALS,
        experiment_name=MLFLOW_EXPERIMENT_TOP,
        model_output_path=f"{MODELS}/xgb_top_users.pkl",
        current_time=today(),
    )

    # df_cost_matrix_results = experiment_cost_matrix(X_train_top, y_train_top)
    # df_cost_matrix_results.to_csv(
    #     "{DATA_PROCESSED}/cost_matrix_experiment_results.csv", index=False
    # )
