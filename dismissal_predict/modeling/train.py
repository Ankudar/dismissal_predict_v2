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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
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
from sklearn.model_selection import cross_val_score, train_test_split
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
N_TRIALS = 200
MLFLOW_EXPERIMENT_MAIN = "xgboost_main_users"
MLFLOW_EXPERIMENT_TOP = "xgboost_top_users"
METRIC_TO_OPTIMIZE = "f1"

target_col = "уволен"

warnings.filterwarnings("ignore")

# Загрузка данных
main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")

logger.info(f"main_users shape: {main_users.shape}")
logger.info(f"top_users shape: {top_users.shape}")


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


def find_best_threshold(y_true, y_probs, cost_fp=1, cost_fn=9):
    thresholds = np.linspace(0, 1, 1000)
    best_score = -np.inf
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape != (2, 2):
            continue  # пропускаем случай с одноклассовыми прогнозами

        tn, fp, fn, tp = cm.ravel()

        # Твоя метрика: FN максимально штрафуется, FP — умеренно
        score = tp + tn - cost_fp * fp - cost_fn * fn

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold


def split_df(data):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(["уволен"], axis=1),
            data["уволен"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=data["уволен"],
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def objective(trial, X_train, y_train, metric="f1"):
    try:
        feature_selector = trial.suggest_categorical(
            "feature_selector", ["xgb_importance", "k_best"]
        )
        top_k = trial.suggest_int("top_k", 5, min(100, X_train.shape[1]))

        if feature_selector == "xgb_importance":
            model_for_selection = XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model_for_selection.fit(X_train, y_train)
            selector = SelectFromModel(
                model_for_selection, threshold=-np.inf, max_features=top_k, prefit=True
            )
        elif feature_selector == "k_best":
            selector = SelectKBest(score_func=f_classif, k=top_k)
            selector.fit(X_train, y_train)

        X_selected = selector.transform(X_train)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.7),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_selected, y_train, cv=5, scoring=metric)
        return scores.mean()
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def run_optuna_experiment(
    X_train, y_train, X_test, y_test, metric, n_trials, experiment_name, model_output_path
):
    try:
        cost_fp = 1
        cost_fn = 9
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        def optuna_objective(trial):
            return objective(trial, X_train, y_train, metric=metric)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        study.optimize(optuna_objective, n_trials=n_trials)

        best_params = study.best_trial.params

        feature_selector = best_params.get("feature_selector", "xgb_importance")
        top_k = best_params.get("top_k", X_train.shape[1])

        if feature_selector == "xgb_importance":
            model_for_selection = XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model_for_selection.fit(X_train, y_train)
            selector = SelectFromModel(
                model_for_selection, threshold=-np.inf, max_features=top_k, prefit=True
            )
        elif feature_selector == "k_best":
            selector = SelectKBest(score_func=f_classif, k=top_k)
            selector.fit(X_train, y_train)

        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()].tolist()

        final_model = XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        final_model.fit(X_train_selected, y_train)
        y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
        best_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= best_threshold).astype(int)

        # Расчет всех метрик
        final_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
        }
        # Метрика, по которой оптимизировали
        final_metric_value = final_metrics.get(metric, final_metrics["f1"])

        # Проверка на существование сохранённой модели
        if os.path.exists(model_output_path):
            loaded = joblib.load(model_output_path)
            existing_model = loaded["model"]
            existing_thresh = find_best_threshold(
                y_test, existing_model.predict_proba(X_test)[:, 1], cost_fp, cost_fn
            )
            existing_proba = existing_model.predict_proba(X_test)[:, 1]
            existing_pred = (existing_proba >= existing_thresh).astype(int)

            existing_metrics = {
                "accuracy": accuracy_score(y_test, existing_pred),
                "precision": precision_score(y_test, existing_pred),
                "recall": recall_score(y_test, existing_pred),
                "roc_auc": roc_auc_score(y_test, existing_proba),
                "f1": f1_score(y_test, existing_pred),
            }

            # Сравнение по кастомному score
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            final_score = tp + tn - cost_fp * fp - cost_fn * fn

            if confusion_matrix(y_test, existing_pred).shape == (2, 2):
                etn, efp, efn, etp = confusion_matrix(y_test, existing_pred).ravel()
                existing_score = etp + etn - cost_fp * efp - cost_fn * efn
            else:
                existing_score = -np.inf

            if final_score > existing_score:
                logger.info("Новая модель лучше сохраненной по кастомной метрике, сохраняем.")
                joblib.dump(
                    {
                        "model": final_model,
                        "threshold": best_threshold,
                        "features": selected_features,
                    },
                    model_output_path,
                )
            else:
                logger.info(
                    "Сохраненная модель лучше по кастомной метрике, оставляем без изменений."
                )
        else:
            logger.info("Сохраняем модель, т.к. других еще не было.")
            joblib.dump(
                {
                    "model": final_model,
                    "threshold": best_threshold,
                    "features": selected_features,
                },
                model_output_path,
            )

        input_example = pd.DataFrame(X_test_selected[:1], columns=selected_features)

        with mlflow.start_run(run_name="final_model"):
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("model_type", "XGBoostClassifier")
            mlflow.log_param("threshold", round(best_threshold, 4))
            mlflow.log_param("optimized_metric", metric)
            mlflow.log_param("feature_selector", feature_selector)
            mlflow.log_param("top_k", top_k)
            mlflow.log_metrics(
                {**final_metrics, f"optimized_metric_value": float(final_metric_value)}
            )
            mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)

            # 1. Confusion matrix
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
            fig_cm.savefig("confusion_matrix.png")
            plt.close(fig_cm)
            mlflow.log_artifact("confusion_matrix.png")

            # 2. ROC curve
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc)
            fig_roc.savefig("roc_curve.png")
            plt.close(fig_roc)
            mlflow.log_artifact("roc_curve.png")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/root6/python/dismissal_predict_v2/mlruns")

    main_users = convert_all_to_float(main_users, exclude_cols=["уволен"])
    top_users = convert_all_to_float(top_users, exclude_cols=["уволен"])
    main_users["уволен"] = (
        pd.to_numeric(main_users["уволен"], errors="coerce").fillna(0).astype(int)
    )
    top_users["уволен"] = pd.to_numeric(top_users["уволен"], errors="coerce").fillna(0).astype(int)

    X_train_main, X_test_main, y_train_main, y_test_main = split_df(main_users)
    run_optuna_experiment(
        X_train_main,
        y_train_main,
        X_test_main,
        y_test_main,
        metric=METRIC_TO_OPTIMIZE,
        n_trials=N_TRIALS,
        experiment_name=MLFLOW_EXPERIMENT_MAIN,
        model_output_path=f"{MODELS}/xgb_main_users.pkl",
    )

    X_train_top, X_test_top, y_train_top, y_test_top = split_df(top_users)
    run_optuna_experiment(
        X_train_top,
        y_train_top,
        X_test_top,
        y_test_top,
        metric=METRIC_TO_OPTIMIZE,
        n_trials=N_TRIALS,
        experiment_name=MLFLOW_EXPERIMENT_TOP,
        model_output_path=f"{MODELS}/xgb_top_users.pkl",
    )
