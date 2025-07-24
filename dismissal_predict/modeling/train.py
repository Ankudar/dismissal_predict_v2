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

# Пути
DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
MODELS = "/home/root6/python/dismissal_predict_v2/models"
os.makedirs(MODELS, exist_ok=True)

INPUT_FILE_MAIN_USERS = f"{DATA_PROCESSED}/main_users_for_train.csv"
INPUT_FILE_TOP_USERS = f"{DATA_PROCESSED}/main_top_for_train.csv"

# Константы
TEST_SIZE = 0.2
RANDOM_STATE = 20
N_TRIALS = 100
MLFLOW_EXPERIMENT_MAIN = "xgboost_main_users"
MLFLOW_EXPERIMENT_TOP = "xgboost_top_users"

TARGET_COL = "уволен"

COST_FP_NUM = 5
COST_FN_NUM = 20


warnings.filterwarnings("ignore")

# Загрузка данных
main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")


def is_new_model_better(new_metrics, old_metrics, metric):
    return new_metrics[metric] > old_metrics.get(metric, 0)


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


def find_best_threshold(y_true, y_probs, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM):
    thresholds = np.linspace(0, 1, 1000)
    best_score = -np.inf
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape != (2, 2):
            continue  # пропускаем случай с одноклассовыми прогнозами

        tn, fp, fn, tp = cm.ravel()
        score = tp + tn - cost_fp * fp - cost_fn * fn

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold


def custom_cv_score(model, X, y, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    threshold = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        y_probs = model.predict_proba(X_valid_fold)[:, 1]

        # 👉 Находим лучший threshold по кастомному скорам
        best_thresh = find_best_threshold(y_valid_fold, y_probs, cost_fp, cost_fn)
        y_preds = (y_probs >= best_thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_valid_fold, y_preds).ravel()
        score = tp + tn - cost_fp * fp - cost_fn * fn
        scores.append(score)
        threshold.append(best_thresh)

    return np.mean(scores), np.mean(threshold)


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


def objective(trial, X_train, y_train):
    try:
        # 👉 вычисляем scale_pos_weight
        counter = Counter(y_train)
        scale_pos_weight = counter[0] / counter[1]

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        score_mean, _ = custom_cv_score(
            model, X_train, y_train, cost_fp=COST_FP_NUM, cost_fn=COST_FN_NUM
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

        def optuna_objective(trial):
            return objective(trial, X_train, y_train)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        manual_optuna_progress(study, n_trials, optuna_objective)

        best_params = study.best_trial.params
        best_mean_score = study.best_value

        final_model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        final_model.fit(X_train, y_train)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]

        best_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= best_threshold).astype(int)

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

        # Проверка метрик
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
                    "threshold": best_threshold,
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
            mlflow.log_param("threshold", round(best_threshold, 4))
            mlflow.log_metric(f"train_{metric}", round(best_mean_score, 3))
            mlflow.log_metrics(
                {
                    **final_metrics,
                    f"optimized_metric_value": round(float(final_metrics[metric]), 3),
                }
            )

            # Логирование модели как артефакта
            mlflow.log_artifact(model_output_path)

            # Логирование модели в MLflow
            mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)  # type: ignore

            # Визуализация матрицы ошибок
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
            fig_cm.savefig("confusion_matrix.png")
            plt.close(fig_cm)
            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")

            # Визуализация ROC кривой
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc)
            fig_roc.savefig("roc_curve.png")
            plt.close(fig_roc)
            mlflow.log_artifact("roc_curve.png")
            os.remove("roc_curve.png")

            # SHAP значения
            explainer = shap.Explainer(final_model)
            shap_values = explainer(X_test)

            shap.initjs()
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
