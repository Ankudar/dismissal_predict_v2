import glob
import logging
import os
import re

from config import MAIN_CONFIGS, Config
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE_CADR = "/home/root6/python/dismissal_predict_v2/data/raw/last_users_from_cadr.xls"
INPUT_FILE_CHILDREN = "/home/root6/python/dismissal_predict_v2/data/raw/children.csv"
INPUT_FILE_DIRECTOR = "/home/root6/python/dismissal_predict_v2/data/raw/director.csv"
INPUT_FILE_MAIN_USERS = "/home/root6/python/dismissal_predict_v2/data/raw/main_users.csv"
INPUT_FILE_STAT = "/home/root6/python/dismissal_predict_v2/data/raw/whisper_stat.csv"
INPUT_ZUP_PATH = "/home/root6/python/dismissal_predict_v2/data/raw/zup"
DATA_INTERIM = "/home/root6/python/dismissal_predict_v2/data/interim"
DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"

CONFIG = Config(MAIN_CONFIGS)
WHISPER_CATEGORIES_WEIGHT = CONFIG.WHISPER_CATEGORIES_WEIGHT


def prepare_data(data):
    try:
        data.columns = data.columns.str.lower()
        data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        data = data.apply(process_spaces)
        data.columns = [replace_spaces(col) for col in data.columns]
        data = drop_duplicated(data)
        return data
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_spaces(s):
    try:
        if isinstance(s, str):
            s = s.strip()
            s = " ".join(s.split())
        return s
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def replace_spaces(s):
    try:
        if isinstance(s, str):
            s = s.strip()
            s = "_".join(s.split())
        return s
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def drop_duplicated(data):
    try:
        num_duplicates = data.duplicated().sum()
        if num_duplicates > 0:
            data = data.drop_duplicates(keep="first").reset_index(drop=True)
        return data
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_last_users_from_cadr(input_file, output_dir):
    try:
        df = pd.read_excel(input_file, header=2)
        df = df[df["планируемая дата выхода из декрета"].isna()]
        columns_to_copy = [
            "дата приема в 1С",
            "Ф.И.О.",
            "Пол",
            "текущая должность на портале",
            "Грейд",
            "КАТЕГОРИЯ",
            "БЕ",
            "ОТДЕЛ",
        ]

        df.columns = df.columns.str.strip()
        missing_columns = [col for col in columns_to_copy if col not in df.columns]
        if missing_columns:
            return

        processed_data = df[columns_to_copy]
        processed_data = prepare_data(processed_data)
        processed_data.rename(columns={"ф.и.о.": "фио"}, inplace=True)

        output_file = os.path.join(output_dir, "check_last_users_update.csv")
        processed_data.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Файл {input_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_children(input_file, output_dir):
    try:
        df = pd.read_csv(input_file, header=0)
        df = df[df["NAME"].str.len() >= 10]

        columns_to_copy = ["USER_ID", "NAME", "DATE_BIRTH", "GENDER"]

        df.columns = df.columns.str.strip()
        missing_columns = [col for col in columns_to_copy if col not in df.columns]
        if missing_columns:
            return

        processed_data = df[columns_to_copy]
        processed_data = df[columns_to_copy].copy()
        processed_data = processed_data.rename(columns={"USER_ID": "id"})
        processed_data = prepare_data(processed_data)

        output_file = os.path.join(output_dir, "children.csv")
        processed_data.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Файл {input_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_director(input_file, output_dir):
    # здесь просто сохраняем дальше
    try:
        df = pd.read_csv(input_file, header=0)

        processed_data = df
        processed_data = prepare_data(processed_data)

        output_file = os.path.join(output_dir, "director.csv")
        processed_data.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Файл {input_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_main_users(input_file, output_dir):
    try:
        df = pd.read_csv(input_file, header=1)
        columns_to_copy = [
            "Дата увольнения",
            "Логин",
            "Должность",
            "Имя",
            "Фамилия",
            "ID",
            "Дата рождения",
        ]

        df.columns = df.columns.str.strip()
        missing_columns = [col for col in columns_to_copy if col not in df.columns]
        if missing_columns:
            return

        processed_data = df[columns_to_copy]
        processed_data = prepare_data(processed_data)

        output_file = os.path.join(output_dir, "main_users.csv")
        processed_data.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Файл {input_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def transform_value(col_name, value):
    try:
        return WHISPER_CATEGORIES_WEIGHT[col_name].get(value, 0)
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_whisper_stat(input_file, output_dir):
    try:
        df = pd.read_csv(input_file, header=0)

        for col in df.columns:
            if col != "логин":
                df[col] = df[col].apply(lambda x: transform_value(col, x))

        df = df.groupby("логин").mean().reset_index()

        output_file = os.path.join(output_dir, "whisper_stat.csv")
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Файл {input_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def process_zup_path(input_dir, output_dir):
    try:
        files = glob.glob(os.path.join(input_dir, "*.xlsx"))
        merged_data = pd.DataFrame()

        for file in files:
            filename = os.path.basename(file)
            month_year = filename.split("_")[1:3]
            month = month_year[0]
            year = "20" + month_year[1]
            new_column_name = f"{month}.{year}"

            data = pd.read_excel(file)

            if len(data.columns) < 4:
                logger.info(f"Ошибка: недостаточно столбцов в {filename}")
                continue

            fio = data.iloc[:, 0]
            values = data.iloc[:, 2]

            grouped_data = data.groupby(fio).agg({values.name: "sum"}).reset_index()
            grouped_data.rename(columns={values.name: new_column_name}, inplace=True)

            if merged_data.empty:
                merged_data = grouped_data
            else:
                merged_data = pd.merge(merged_data, grouped_data, on=fio.name, how="outer")

        for col in merged_data.columns[1:]:
            merged_data[col] = (
                pd.to_numeric(merged_data[col], errors="coerce").fillna(0).astype(float)
            )

        date_columns = sorted(
            merged_data.columns[1:],
            key=lambda x: pd.to_datetime(f"{x.split('.')[1]}-{x.split('.')[0]}-01"),
        )
        sorted_columns = ["Месяц"] + date_columns
        merged_data = merged_data[sorted_columns]

        result_data = pd.DataFrame()
        result_data["ФИО"] = merged_data["Месяц"]
        result_data["ср_зп"] = round(merged_data.iloc[:, 1:].mean(axis=1), 2)

        unique_fios = result_data["ФИО"].unique()
        pattern = re.compile(r"^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+")
        filtered_fios = [fio for fio in unique_fios if pattern.match(fio)]
        processed_data = result_data[result_data["ФИО"].isin(filtered_fios)]
        processed_data = processed_data.reset_index(drop=True)

        num_categories = 5
        processed_data["уровень_зп"] = pd.qcut(
            processed_data["ср_зп"],
            q=num_categories,
            labels=["низкий", "ниже среднего", "средний", "выше среднего", "высокий"],
        )

        processed_data = prepare_data(processed_data)
        output_file = os.path.join(output_dir, "zup.csv")
        processed_data.to_csv(output_file, index=False)
        logger.info(f"Файл {output_file} успешно обработан и сохранен.")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def run_all():
    process_last_users_from_cadr(INPUT_FILE_CADR, DATA_INTERIM)
    process_children(INPUT_FILE_CHILDREN, DATA_INTERIM)
    process_director(INPUT_FILE_DIRECTOR, DATA_INTERIM)
    process_main_users(INPUT_FILE_MAIN_USERS, DATA_INTERIM)
    process_whisper_stat(INPUT_FILE_STAT, DATA_INTERIM)
    process_zup_path(INPUT_ZUP_PATH, DATA_INTERIM)


if __name__ == "__main__":
    run_all()
