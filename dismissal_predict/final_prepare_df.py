from datetime import datetime
import logging
import re

import joblib
import numpy as np
import pandas as pd
from rusgenderdetection import get_gender  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_INTERIM = "/home/root6/python/dismissal_predict_v2/data/interim"
DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
INPUT_FILE_MAIN_USERS = f"{DATA_INTERIM}/main_users.csv"
INPUT_FILE_CADR = f"{DATA_INTERIM}/check_last_users_update.csv"
INPUT_FILE_CHILDREN = f"{DATA_INTERIM}/children.csv"
INPUT_FILE_STAT = f"{DATA_INTERIM}/whisper_stat.csv"
INPUT_ZUP_PATH = f"{DATA_INTERIM}/zup.csv"

DROP_COLS = [
    "id",
    "логин",
    "имя",
    "фамилия",
    "фио",
    "должность",
    "дата_увольнения",
    "дата_рождения",
    "дата_приема_в_1с",
    "текущая_должность_на_портале",
    "отдел",
    "уровень_зп",
    "грейд",
]

FLOAT_COLS = ["тон", "увольнение", "оффер", "вредительство", "личная жизнь", "стресс", "конфликты"]

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
users_cadr = pd.read_csv(INPUT_FILE_CADR, delimiter=",", decimal=",")
users_salary = pd.read_csv(INPUT_ZUP_PATH, delimiter=",", decimal=",")
children = pd.read_csv(INPUT_FILE_CHILDREN, delimiter=",", decimal=",")
stat = pd.read_csv(INPUT_FILE_STAT, delimiter=",", decimal=",")


class DataPreprocessor:
    def __init__(self):
        self.cat_cols = []
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.scaler = None
        self.numeric_cols = []
        self.preprocessor = None

    def save(self, path_prefix):
        joblib.dump(self.ordinal_encoder, f"{path_prefix}_ordinal_encoder.pkl")
        joblib.dump(self.onehot_encoder, f"{path_prefix}_onehot_encoder.pkl")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(self.cat_cols, f"{path_prefix}_cat_cols.pkl")
        joblib.dump(self.numeric_cols, f"{path_prefix}_numeric_cols.pkl")
        joblib.dump(self.preprocessor, f"{path_prefix}_preprocessor.pkl")

    def load(self, path_prefix):
        self.ordinal_encoder = joblib.load(f"{path_prefix}_ordinal_encoder.pkl")
        self.onehot_encoder = joblib.load(f"{path_prefix}_onehot_encoder.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.cat_cols = joblib.load(f"{path_prefix}_cat_cols.pkl")
        self.numeric_cols = joblib.load(f"{path_prefix}_numeric_cols.pkl")
        self.preprocessor = joblib.load(f"{path_prefix}_preprocessor.pkl")

    def _handle_nans(self, df):
        for col in self.numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        # Сохраняем столбец 'уволен' для последующего добавления
        if "уволен" in df.columns:
            uvolen_series = df["уволен"]
            df.drop(columns=["уволен"], inplace=True)

        # Категориальные признаки
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in self.cat_cols:
            df[col] = df[col].astype(str).fillna("other")

        # Числовые признаки
        self.numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        df = self._handle_nans(df)

        # Разделение категориальных признаков на OneHot и Ordinal
        onehot_cols = [col for col in self.cat_cols if df[col].nunique() <= 10]
        ordinal_cols = [col for col in self.cat_cols if df[col].nunique() >= 11]

        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", self.onehot_encoder, onehot_cols),
                ("ordinal", self.ordinal_encoder, ordinal_cols),
                ("num", MinMaxScaler(), self.numeric_cols),
            ],
            remainder="passthrough",
        )

        df_transformed = self.preprocessor.fit_transform(df)
        df_transformed = pd.DataFrame(
            df_transformed, columns=self.preprocessor.get_feature_names_out()  # type: ignore
        )

        # Добавляем столбец 'уволен' обратно
        if "уволен" in df.columns:
            df_transformed["уволен"] = uvolen_series.values

        return df_transformed

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        # Сохраняем столбец 'уволен' для последующего добавления
        if "уволен" in df.columns:
            uvolen_series = df["уволен"]
            df.drop(columns=["уволен"], inplace=True)

        # Убедимся, что все необходимые категориальные колонки присутствуют
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "other"
            df[col] = df[col].astype(str).fillna("other")

        # Обрабатываем числовые признаки
        df = self._handle_nans(df)

        # Преобразуем категориальные признаки
        df_transformed = self.preprocessor.transform(df)  # type: ignore
        df_transformed = pd.DataFrame(
            df_transformed, columns=self.preprocessor.get_feature_names_out()  # type: ignore
        )

        # Добавляем столбец 'уволен' обратно
        if "уволен" in df.columns:
            df_transformed["уволен"] = uvolen_series.values

        return df_transformed


def merge_base(bases, index, merge_type):
    try:
        merged_df = bases[0]
        for base in bases[1:]:
            merged_df = pd.merge(merged_df, base, on=index, how=merge_type)
        return merged_df
    except Exception as e:
        logger.info(f"Ошибка при объединении: {e}")
        raise


def convert_dates(df):
    try:
        # Форматы, которые могут встречаться в данных
        date_columns = ["дата_рождения", "дата_увольнения", "дата_приема_в_1с"]
        for col in date_columns:
            df[col] = pd.to_datetime(
                df[col], errors="coerce"
            )  # Автоматическое определение формата

        return df
    except Exception as e:
        logger.info(f"Ошибка при конвертации дат: {e}")
        raise


def mode_with_tie(series):
    try:
        if series.empty:
            return "none"

        counts = series.value_counts()
        m_count = counts.get("m", 0)
        f_count = counts.get("f", 0)

        if m_count > f_count:
            return "m"
        elif f_count > m_count:
            return "f"
        else:
            return "mf"
    except Exception as e:
        logger.info(f"Ошибка в mode_with_tie: {e}")
        raise


def extract_first_name(full_name):
    # Проверка на NaN
    if pd.isna(full_name):
        return ""
    parts = full_name.split()
    return parts[1] if len(parts) > 1 else ""


def determine_gender(full_name):
    first_name = extract_first_name(full_name)
    if first_name:
        gender = get_gender(first_name)
        if gender == 0:
            return "женский"
        elif gender == 1:
            return "мужской"
        else:
            return "неизвестно"
    return "неизвестно"


def main_prepare_for_all(main_users, users_salary, users_cadr, children):
    try:
        today = pd.to_datetime(datetime.now().date())
        children["date_birth"] = pd.to_datetime(
            children["date_birth"], format="%d.%m.%Y", errors="coerce"
        )
        children["age"] = ((datetime.now() - children["date_birth"]).dt.days / 365).round(1)

        grouped_children = (
            children.groupby("id")
            .agg(
                child_num=("name", "count"),
                avg_child_age=("age", "mean"),
                main_child_gender=("gender", mode_with_tie),
            )
            .reset_index()
        )

        for df in [users_salary, users_cadr]:
            df["фио"] = df["фио"].str.split().str[:2].str.join(" ")

        main_users["фио"] = main_users["фамилия"] + " " + main_users["имя"]
        main_users = merge_base([main_users, users_cadr], "фио", "left")
        main_users = merge_base([main_users, users_salary], "фио", "left")
        main_users = merge_base([main_users, grouped_children], "id", "left")

        logins_to_remove = [
            "root24",
            "root35",
            "root36",
            "test.testovich",
            "система",
            "admin",
            "двадцать тест",
        ]
        main_users = main_users[~main_users["логин"].isin(logins_to_remove)]
        main_users["пол"] = main_users["фио"].apply(determine_gender)
        main_users.replace("nan", pd.NA, inplace=True)
        main_users = main_users.dropna(subset=["имя", "фамилия"])
        main_users = convert_dates(main_users)

        for col in ["логин", "должность", "имя", "фамилия"]:
            main_users[col] = main_users[col].astype(str)
        main_users["ср_зп"] = main_users["ср_зп"].astype(float)
        main_users["уволен"] = main_users["дата_увольнения"].notna().astype(int)

        main_users["дата_увольнения"] = pd.to_datetime(
            main_users["дата_увольнения"], errors="coerce"
        )

        main_users["возраст"] = np.where(
            main_users["дата_рождения"].notna(),
            (today - main_users["дата_рождения"]).dt.days // 365,  # type: ignore
            np.nan,
        )

        main_users["стаж"] = np.where(
            main_users["дата_увольнения"].notna(),
            (main_users["дата_увольнения"] - main_users["дата_приема_в_1с"]).dt.days / 365,
            (today - main_users["дата_приема_в_1с"]).dt.days / 365,  # type: ignore
        )

        main_users["стаж"] = np.maximum(main_users["стаж"], 0)

        non_null_positions = main_users["текущая_должность_на_портале"].dropna().unique()
        position_to_num = {position: idx + 1 for idx, position in enumerate(non_null_positions)}
        main_users["текущая_должность_на_портале_num"] = main_users[
            "текущая_должность_на_портале"
        ].map(position_to_num)

        non_null_positions = main_users["отдел"].dropna().unique()
        position_to_num = {position: idx + 1 for idx, position in enumerate(non_null_positions)}
        main_users["отдел_num"] = main_users["отдел"].map(position_to_num)

        main_users.to_csv(f"{DATA_PROCESSED}/main_all.csv", index=False)

        preprocessor = DataPreprocessor()
        main_users_for_train = preprocessor.fit(main_users)
        main_users_for_train.to_csv(f"{DATA_PROCESSED}/main_users_for_train.csv", index=False)

        if "уволен" in main_users.columns:
            main_users_for_train["уволен"] = main_users["уволен"].values

        main_users_for_train.to_csv(f"{DATA_PROCESSED}/main_users_for_train.csv", index=False)

        preprocessor.save(f"{DATA_PROCESSED}/preprocessor")
        print(
            "NaNs in main_users_for_train:\n",
            main_users_for_train.isnull().sum()[main_users_for_train.isnull().any()],
        )
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def prepare_with_mic():
    main_all = pd.read_csv(f"{DATA_PROCESSED}/main_all.csv", delimiter=",", decimal=",")
    main_top = merge_base([stat, main_all], "логин", "left")

    logins_to_remove = [
        "root24",
        "root35",
        "root36",
        "test.testovich",
        "система",
        "admin",
        "двадцать тест",
    ]
    main_top = main_top[~main_top["логин"].isin(logins_to_remove)]

    for col in FLOAT_COLS:
        if col in main_top.columns:
            main_top[col] = pd.to_numeric(main_top[col], errors="coerce")
            main_top[col] = main_top[col].fillna(main_top[col].median())

    main_top.to_csv(f"{DATA_PROCESSED}/main_top.csv", index=False)

    preprocessor = DataPreprocessor()
    preprocessor.load(f"{DATA_PROCESSED}/preprocessor")

    if "уволен" in main_top.columns:
        uvolen_series = main_top["уволен"].astype(int)  # Преобразуем в int

    main_top_for_train = preprocessor.transform(main_top)

    if "уволен" in main_top.columns:
        main_top_for_train["уволен"] = uvolen_series.values

    main_top_for_train.to_csv(f"{DATA_PROCESSED}/main_top_for_train.csv", index=False)
    print(
        "NaNs in main_top_for_train:\n",
        main_top_for_train.isnull().sum()[main_top_for_train.isnull().any()],
    )


def run_all():
    main_prepare_for_all(main_users, users_salary, users_cadr, children)
    prepare_with_mic()


if __name__ == "__main__":
    logger.info("Финальная подготовка баз началась")
    run_all()
    logger.info("Финальная подготовка баз завершилась")
