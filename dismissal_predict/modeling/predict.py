from datetime import datetime
import logging
import os
import sys

import joblib
import pandas as pd
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dismissal_predict import DROP_COLS, FLOAT_COLS, DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Пути
DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
DATA_INTERIM = "/home/root6/python/dismissal_predict_v2/data/interim"
RESULTS_DIR = "/home/root6/python/dismissal_predict_v2/data/results"
MODELS_DIR = "/home/root6/python/dismissal_predict_v2/models"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Файлы
INPUT_FILE_MAIN_ALL = os.path.join(DATA_PROCESSED, "main_all.csv")
INPUT_FILE_MAIN_TOP = os.path.join(DATA_PROCESSED, "main_top.csv")
INPUT_FILE_CADR = os.path.join(DATA_INTERIM, "check_last_users_update.csv")
MODEL_MAIN = os.path.join(MODELS_DIR, "main_users.pkl")
MODEL_TOP = os.path.join(MODELS_DIR, "top_users.pkl")
PREPROCESSOR_MAIN_PATH = os.path.join(DATA_PROCESSED, "preprocessor.pkl")
PREPROCESSOR_TOP_PATH = os.path.join(DATA_PROCESSED, "preprocessor_top.pkl")

# Данные
df_cadr = pd.read_csv(INPUT_FILE_CADR, delimiter=",", decimal=",")
df_main_all = pd.read_csv(INPUT_FILE_MAIN_ALL, delimiter=",", decimal=",")
df_main_top = pd.read_csv(INPUT_FILE_MAIN_TOP, delimiter=",", decimal=",")


def load_model_and_threshold(model_path):
    model_bundle = joblib.load(model_path)
    return (
        model_bundle["model"],
        model_bundle["threshold"],
        model_bundle.get("selected_features", None),
    )


def update_results_with_cadr(result_df, main_df):
    result_df["уволен"] = 0  # по умолчанию считаем всех не уволенными

    # шаг 1 — установить по main_df
    for fio in result_df["фио"]:
        if fio in main_df["фио"].values:
            is_fired = main_df.loc[main_df["фио"] == fio, "уволен"].values[0]
            result_df.loc[result_df["фио"] == fio, "уволен"] = int(float(is_fired))

    # шаг 2 — если сотрудника нет в кадровом df_cadr (ФИО в виде "Фамилия Имя Отчество")
    cadr_fio_short = (
        df_cadr["фио"].dropna().apply(lambda x: " ".join(x.strip().split()[:2])).unique()
    )
    for fio in result_df["фио"]:
        if fio not in cadr_fio_short:
            result_df.loc[result_df["фио"] == fio, "уволен"] = 1

    return result_df


def add_predictions_to_excel(original_df, model, threshold, result_file, preprocessor, features):
    try:
        original_df["уволен"] = original_df["дата_увольнения"].notna().astype(int)

        if "фио" not in original_df.columns:
            raise ValueError("В датафрейме отсутствует колонка 'фио'")

        display_df = original_df[["фио", "уволен"]].copy()
        predict_df = original_df[original_df["дата_увольнения"].isna()].copy()

        predict_df_clean = predict_df.drop(
            columns=[col for col in DROP_COLS if col in predict_df.columns]
        )
        predict_clean = preprocessor.transform(predict_df_clean)
        predict_clean = predict_clean.apply(pd.to_numeric, errors="coerce").fillna(0)
        predict_clean = predict_clean.drop(columns=["уволен"], errors="ignore")

        if features:
            predict_clean = predict_clean[features]

        probabilities = model.predict_proba(predict_clean)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        today = datetime.today().strftime("%d.%m.%Y")

        result_today = pd.DataFrame(
            {
                "фио": predict_df["фио"].values,
                "предсказание_увольнения": predictions,
                today: probabilities,
            }
        )

        # 🔹 SHAP: рассчитываем и сохраняем
        explainer = shap.Explainer(model)
        shap_values = explainer(predict_clean)

        # Проверка на пустой список признаков
        if not features:
            features = list(predict_clean.columns)

        # Обработка SHAP значений для модели с вероятностями
        if shap_values.values.ndim == 3:
            # Если 3D — берём SHAP для класса 1
            shap_vals_for_class1 = shap_values.values[:, :, 1]
        elif shap_values.values.ndim == 2:
            shap_vals_for_class1 = shap_values.values
        else:
            raise ValueError(f"Неожиданный формат SHAP-значений: ndim={shap_values.values.ndim}")

        shap_df = pd.DataFrame(shap_vals_for_class1, columns=features)
        shap_df.insert(0, "фио", predict_df["фио"].values)

        shap_path = result_file.replace(".xlsx", "_shap.csv")
        shap_df.to_csv(shap_path, index=False, encoding="utf-8-sig")
        logger.info(f"SHAP значения сохранены в {shap_path}")

        # 🔹 Обновляем Excel
        if os.path.exists(result_file):
            final_df = pd.read_excel(result_file)
            final_df = update_results_with_cadr(final_df, original_df)
        else:
            final_df = display_df.copy()
            final_df["предсказание_увольнения"] = None
            final_df = update_results_with_cadr(final_df, original_df)

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
                final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)

        final_df.to_excel(result_file, index=False)
        logger.info(f"Результаты предсказаний успешно записаны в {result_file}")

    except Exception as e:
        logger.error(f"Ошибка при добавлении предсказаний в Excel: {e}")
        raise


if __name__ == "__main__":
    logger.info("Начало выполнения предсказаний")

    try:
        # 🔹 Загружаем отдельные препроцессоры
        preprocessor_main = DataPreprocessor.load(PREPROCESSOR_MAIN_PATH)
        preprocessor_top = DataPreprocessor.load(PREPROCESSOR_TOP_PATH)

        model_main, threshold_main, features_main = load_model_and_threshold(MODEL_MAIN)
        model_top, threshold_top, features_top = load_model_and_threshold(MODEL_TOP)

        if not features_main:
            features_main = [
                col for col in df_main_all.columns if col != "уволен" and col not in DROP_COLS
            ]

        if not features_top:
            features_top = [
                col for col in df_main_top.columns if col != "уволен" and col not in DROP_COLS
            ]

        # Чистим и нормализуем данные
        df_main_all_clean = df_main_all.drop(
            columns=[col for col in DROP_COLS if col in df_main_all.columns]
        )
        df_main_top_clean = df_main_top.drop(
            columns=[col for col in DROP_COLS if col in df_main_top.columns]
        )

        for col in FLOAT_COLS:
            if col in df_main_all_clean.columns:
                df_main_all_clean[col] = pd.to_numeric(df_main_all_clean[col], errors="coerce")
                df_main_all_clean[col] = df_main_all_clean[col].fillna(
                    df_main_all_clean[col].median()
                )
            if col in df_main_top_clean.columns:
                df_main_top_clean[col] = pd.to_numeric(df_main_top_clean[col], errors="coerce")
                df_main_top_clean[col] = df_main_top_clean[col].fillna(
                    df_main_top_clean[col].median()
                )

        # 🔹 Преобразование с разными препроцессорами
        df_main_all_proc = preprocessor_main.transform(df_main_all_clean)
        df_main_top_proc = preprocessor_top.transform(df_main_top_clean)

        features_main = [f for f in features_main if f in df_main_all_proc.columns]
        features_top = [f for f in features_top if f in df_main_top_proc.columns]

        df_main_all_proc = (
            df_main_all_proc[features_main + ["уволен"]]
            if "уволен" in df_main_all_proc.columns
            else df_main_all_proc[features_main]
        )
        df_main_top_proc = (
            df_main_top_proc[features_top + ["уволен"]]
            if "уволен" in df_main_top_proc.columns
            else df_main_top_proc[features_top]
        )

        # 🔹 Предсказания
        add_predictions_to_excel(
            df_main_all.copy(),
            model_main,
            threshold_main,
            os.path.join(RESULTS_DIR, "result_all.xlsx"),
            preprocessor_main,
            features_main,
        )

        add_predictions_to_excel(
            df_main_top.copy(),
            model_top,
            threshold_top,
            os.path.join(RESULTS_DIR, "result_top.xlsx"),
            preprocessor_top,
            features_top,
        )

    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {e}")
