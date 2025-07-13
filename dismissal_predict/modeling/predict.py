from datetime import datetime
import logging
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dismissal_predict import DROP_COLS, FLOAT_COLS, DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
RESULTS_DIR = "/home/root6/python/dismissal_predict_v2/data/results"
MODELS_DIR = "/home/root6/python/dismissal_predict_v2/models"
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE_MAIN_ALL = os.path.join(DATA_PROCESSED, "main_all.csv")
INPUT_FILE_MAIN_TOP = os.path.join(DATA_PROCESSED, "main_top.csv")
MODEL_MAIN = os.path.join(MODELS_DIR, "xgb_main_users.pkl")
MODEL_TOP = os.path.join(MODELS_DIR, "xgb_top_users.pkl")
PREPROCESSOR_PATH = os.path.join(DATA_PROCESSED, "preprocessor")


def load_model_and_threshold(model_path):
    model_bundle = joblib.load(model_path)
    return model_bundle["model"], model_bundle["threshold"], model_bundle.get("features", None)


def add_predictions_to_excel(original_df, model, threshold, result_file, preprocessor, features):
    try:
        # Обновляем колонку "уволен"
        original_df["уволен"] = original_df["дата_увольнения"].notna().astype(int)

        if "фио" not in original_df.columns:
            raise ValueError("В датафрейме отсутствует колонка 'фио'")

        display_df = original_df[["фио", "уволен"]].copy()
        predict_df = original_df[original_df["дата_увольнения"].isna()].copy()

        # Удаляем DROP_COLS перед трансформацией
        predict_df_clean = predict_df.drop(
            columns=[col for col in DROP_COLS if col in predict_df.columns]
        )

        # Применяем препроцессинг
        predict_clean = preprocessor.transform(predict_df_clean)

        # Приведение к числовым типам
        predict_clean = predict_clean.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Удалим колонку "уволен", если она случайно попала
        predict_clean = predict_clean.drop(columns=["уволен"], errors="ignore")

        # ❗️Оставляем только те признаки, которые использовала модель
        if features:
            predict_clean = predict_clean[features]

        probabilities = model.predict_proba(predict_clean)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        # Текущая дата
        today = datetime.today().strftime("%d.%m.%Y")

        # DataFrame с результатами
        result_today = pd.DataFrame(
            {
                "фио": predict_df["фио"].values,
                today: probabilities,
                "предсказание_увольнения": predictions,
            }
        )

        # Загружаем или создаём финальный файл
        if os.path.exists(result_file):
            final_df = pd.read_excel(result_file)
        else:
            final_df = display_df.copy()
            final_df["предсказание_увольнения"] = None

        # Обновляем или добавляем строки
        for _, row in result_today.iterrows():
            fio = row["фио"]
            if fio in final_df["фио"].values:
                final_df.loc[final_df["фио"] == fio, today] = row[today]
                final_df.loc[final_df["фио"] == fio, "предсказание_увольнения"] = row[
                    "предсказание_увольнения"
                ]
            else:
                new_row = {
                    "фио": fio,
                    "уволен": 0,
                    "предсказание_увольнения": row["предсказание_увольнения"],
                    today: row[today],
                }
                final_df = final_df._append(new_row, ignore_index=True)

        final_df.to_excel(result_file, index=False)
        logger.info(f"Результаты предсказаний успешно записаны в {result_file}")

    except Exception as e:
        logger.error(f"Ошибка при добавлении предсказаний в Excel: {e}")
        raise


if __name__ == "__main__":
    df_main_all = pd.read_csv(INPUT_FILE_MAIN_ALL, delimiter=",", decimal=",")
    df_main_top = pd.read_csv(INPUT_FILE_MAIN_TOP, delimiter=",", decimal=",")

    preprocessor = DataPreprocessor()
    preprocessor.load(PREPROCESSOR_PATH)

    model_main, threshold_main, features_main = load_model_and_threshold(MODEL_MAIN)
    if not features_main:
        features_main = [
            col for col in df_main_all.columns if col != "уволен" and col not in DROP_COLS
        ]

    model_top, threshold_top, features_top = load_model_and_threshold(MODEL_TOP)
    if not features_top:
        features_top = [
            col for col in df_main_top.columns if col != "уволен" and col not in DROP_COLS
        ]

    df_main_all_clean = df_main_all.drop(
        columns=[col for col in DROP_COLS if col in df_main_all.columns]
    )
    df_main_top_clean = df_main_top.drop(
        columns=[col for col in DROP_COLS if col in df_main_top.columns]
    )

    for col in FLOAT_COLS:
        if col in df_main_top_clean.columns:
            df_main_top_clean[col] = pd.to_numeric(df_main_top_clean[col], errors="coerce")
            df_main_top_clean[col] = df_main_top_clean[col].fillna(df_main_top_clean[col].median())

    df_main_all_proc = preprocessor.transform(df_main_all_clean)
    features_main = [f for f in features_main if f in df_main_all_proc.columns]
    if "уволен" in df_main_all_proc.columns:
        df_main_all_proc = df_main_all_proc[features_main + ["уволен"]]
    else:
        df_main_all_proc = df_main_all_proc[features_main]

    df_main_top_proc = preprocessor.transform(df_main_top_clean)
    features_top = [f for f in features_top if f in df_main_top_proc.columns]
    df_main_top_proc = (
        df_main_top_proc[features_top + ["уволен"]]
        if "уволен" in df_main_top_proc
        else df_main_top_proc[features_top]
    )

    add_predictions_to_excel(
        df_main_all.copy(),
        model_main,
        threshold_main,
        os.path.join(RESULTS_DIR, "result_all.xlsx"),
        preprocessor,
        features_main,
    )

    add_predictions_to_excel(
        df_main_top.copy(),
        model_top,
        threshold_top,
        os.path.join(RESULTS_DIR, "result_top.xlsx"),
        preprocessor,
        features_top,
    )
