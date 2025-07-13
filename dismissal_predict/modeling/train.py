import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

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
METRIC_TO_OPTIMIZE = "f1"

warnings.filterwarnings("ignore")

# Загрузка данных
main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
top_users = pd.read_csv(INPUT_FILE_TOP_USERS, delimiter=",", decimal=",")


def convert_to_numeric(data):
    for col in data.columns:
        data[col] = pd.to_numeric(data[col])
    return data


def split_df(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(["уволен"], axis=1),
        data["уволен"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["уволен"],
    )
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)

    # Кросс-валидация
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    return scores.mean()


def run_optuna_experiment(
    X_train, y_train, X_test, y_test, metric, n_trials, experiment_name, model_output_path
):
    mlflow.set_experiment(experiment_name)

    def optuna_objective(trial):
        return objective(trial, X_train, y_train)

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials)

    best_params = study.best_trial.params
    final_model = XGBClassifier(
        **best_params, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss"
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Вероятности для положительного класса
    final_accuracy = accuracy_score(y_test, y_pred)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
    final_f1 = f1_score(y_test, y_pred)

    # Проверка на существование сохранённой модели
    if os.path.exists(model_output_path):
        existing_model = joblib.load(model_output_path)
        existing_pred = existing_model.predict(X_test)
        existing_f1 = f1_score(y_test, existing_pred)
        existing_accuracy = accuracy_score(y_test, y_pred)
        existing_precision = precision_score(y_test, y_pred)
        existing_recall = recall_score(y_test, y_pred)
        existing_roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Сравнение метрик
        if final_accuracy > existing_accuracy:
            print("Новая модель лучше сохраненной, сохраняем.")
            joblib.dump(final_model, model_output_path)
        else:
            print("Сохраненная модель лучше полученной, оставляем без изменений.")
    else:
        print("Сохраняем модель, т.к. других еще не было.")
        joblib.dump(final_model, model_output_path)

    input_example = X_test.iloc[0].to_dict()

    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {
                "f1": float(final_f1),
                "accuracy": float(final_accuracy),
                "precision": float(final_precision),
                "recall": float(final_recall),
                "roc_auc": float(final_roc_auc),
            }
        )
        mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)


if __name__ == "__main__":
    main_users = convert_to_numeric(main_users)
    top_users = convert_to_numeric(top_users)

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
