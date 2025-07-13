from datetime import datetime
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

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
        self.scaler = None
        self.target_col = "уволен"
        self.numeric_cols = []

    def save(self, path_prefix):
        joblib.dump(self.ordinal_encoder, f"{path_prefix}_ordinal_encoder.pkl")
        joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(self.cat_cols, f"{path_prefix}_cat_cols.pkl")
        joblib.dump(self.numeric_cols, f"{path_prefix}_numeric_cols.pkl")

    def load(self, path_prefix):
        self.ordinal_encoder = joblib.load(f"{path_prefix}_ordinal_encoder.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.cat_cols = joblib.load(f"{path_prefix}_cat_cols.pkl")
        self.numeric_cols = joblib.load(f"{path_prefix}_numeric_cols.pkl")

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        # Категориальные признаки
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in self.cat_cols:
            df[col] = df[col].astype(str).fillna("other")

        # Числовые признаки
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(df[col].median())

        # Ordinal encoding
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[self.cat_cols] = self.ordinal_encoder.fit_transform(df[self.cat_cols])

        # Масштабирование числовых признаков (кроме целевой переменной)
        self.numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)

        self.scaler = StandardScaler()
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])

        # Удаляем NaN, если остались
        if df.isnull().any().any():
            df = df.fillna(0)

        return df

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        # Убедимся, что все необходимые категориальные колонки присутствуют
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "other"
            df[col] = df[col].astype(str).fillna("other")

        # Обрабатываем числовые признаки
        for col in self.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan

        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(df[col].median())

        # Преобразуем категориальные признаки
        df[self.cat_cols] = self.ordinal_encoder.transform(df[self.cat_cols])

        # Масштабируем числовые признаки
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        # Удаляем NaN, если остались
        if df.isnull().any().any():
            df = df.fillna(0)

        return df


def merge_base(bases, index, merge_type):
    try:
        merged_df = bases[0]
        for base in bases[1:]:
            merged_df = pd.merge(merged_df, base, on=index, how=merge_type)

        return merged_df
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def convert_dates(df):
    try:
        target_format = "%d.%m.%Y %H:%M:%S"

        df["дата_рождения"] = pd.to_datetime(
            df["дата_рождения"], format="%d.%m.%Y", errors="coerce"
        )
        df["дата_приема_в_1с"] = pd.to_datetime(
            df["дата_приема_в_1с"], format="%Y-%m-%d", errors="coerce"
        )

        date_columns = ["дата_увольнения", "дата_рождения", "дата_приема_в_1с"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format=target_format, errors="coerce")

        return df
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


# def mode_with_tie(series):
#     try:
#         if series.empty:
#             return "none"

#         counts = series.value_counts()
#         max_count = counts.max()
#         modes = counts[counts == max_count].index.tolist()

#         if "m/f" in modes:
#             modes = ["m/f" if mode == "f/m" else mode for mode in modes]
#         return "/".join(set(modes))
#     except Exception as e:
#         logger.info(f"Ошибка: {e}")
#         raise


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
                main_child_gender=(
                    "gender",
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan,
                ),
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
        main_users["пол"] = main_users["пол"].fillna("unknown")
        main_users.replace("nan", pd.NA, inplace=True)
        main_users = main_users.dropna(subset=["имя", "фамилия"])
        main_users = convert_dates(main_users)

        for col in ["логин", "должность", "имя", "фамилия"]:
            main_users[col] = main_users[col].astype(str)
        main_users["ср_зп"] = main_users["ср_зп"].astype(float)
        main_users["уволен"] = main_users["дата_увольнения"].notna().astype(int)

        main_users["возраст"] = np.where(
            main_users["дата_рождения"].notna(),
            (today - main_users["дата_рождения"]).dt.days // 365,
            np.nan,
        )
        main_users["стаж"] = np.where(
            main_users["дата_увольнения"].notna(),
            (main_users["дата_увольнения"] - main_users["дата_приема_в_1с"]).dt.days / 365,
            (today - main_users["дата_приема_в_1с"]).dt.days / 365,
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

    main_top_for_train = preprocessor.transform(main_top)
    main_top_for_train.to_csv(f"{DATA_PROCESSED}/main_top_for_train.csv", index=False)
    print(
        "NaNs in main_top_for_train:\n",
        main_top_for_train.isnull().sum()[main_top_for_train.isnull().any()],
    )


def run_all():
    main_prepare_for_all(main_users, users_salary, users_cadr, children)
    prepare_with_mic()


if __name__ == "__main__":
    logger.info(f"Финальная подготовка баз началась")
    run_all()
    logger.info(f"Финальная подготовка баз завершилась")
