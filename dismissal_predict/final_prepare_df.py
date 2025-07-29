from datetime import datetime
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rusgenderdetection import get_gender  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_INTERIM = "/home/root6/python/dismissal_predict_v2/data/interim"
DATA_PROCESSED = "/home/root6/python/dismissal_predict_v2/data/processed"
INPUT_FILE_MAIN_USERS = f"{DATA_INTERIM}/main_users.csv"
INPUT_FILE_CADR = f"{DATA_INTERIM}/check_last_users_update.csv"
INPUT_HISTORY_CADR = f"{DATA_INTERIM}/history_cadr_base.csv"
INPUT_FILE_CHILDREN = f"{DATA_INTERIM}/children.csv"
INPUT_FILE_DIRECTOR = f"{DATA_INTERIM}/director.csv"
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
    "id_руководителя",
]

FLOAT_COLS = ["тон", "увольнение", "оффер", "вредительство", "личная жизнь", "стресс", "конфликты"]

LOGINS_TO_REMOVE = [
    "root24",
    "root35",
    "root36",
    "test.testovich",
    "admin",
    "ib_tech",
    "ib_tech2",
    "lab.s10",
    "lab.s12",
    "lab.s13",
    "lab.s3",
    "lab.t20",
    "test.ap",
    "test_top",
    "test7",
    "testacc",
    "testacc1",
    "testacc12",
    "testacc14",
    "testacc16",
    "testacc17",
    "testacc18",
    "testacc19",
    "testacc2",
    "testacc20",
    "testacc21",
    "testacc22",
    "testacc3",
    "testacc4",
    "testacc5",
    "testacc7",
    "testacc9",
    "testacc10",
    "testacc11",
    "teacher.eng.11",
    "wf_jurist",
    "wf_sale",
    "wf_sale2",
    "система",
]

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
users_cadr = pd.read_csv(INPUT_FILE_CADR, delimiter=",", decimal=",")
history_cadr = pd.read_csv(INPUT_HISTORY_CADR, delimiter=",", decimal=",")
users_salary = pd.read_csv(INPUT_ZUP_PATH, delimiter=",", decimal=",")
children = pd.read_csv(INPUT_FILE_CHILDREN, delimiter=",", decimal=",")
director = pd.read_csv(INPUT_FILE_DIRECTOR, delimiter=",", decimal=",")
stat = pd.read_csv(INPUT_FILE_STAT, delimiter=",", decimal=",")


director = director[["id", "id_руководителя"]].copy()


class DataPreprocessor:
    def __init__(self):
        self.cat_cols = []
        self.ordinal_encoder = None
        self.onehot_encoder = None
        self.scaler = None
        self.numeric_cols = []
        self.preprocessor = None

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def drop_trash_feature(self, df, threshold=0.9):
        high_nan_cols = df.columns[df.isnull().mean() > threshold].tolist()
        if high_nan_cols:
            print(f"Удалены признаки с большим % NaN: {high_nan_cols}")
            df = df.drop(columns=high_nan_cols)
        return df

    # def drop_trash_rows(self, df, threshold=0.5):
    #     row_nan_fraction = df.isnull().mean(axis=1)
    #     bad_rows = df.index[row_nan_fraction > threshold]
    #     if len(bad_rows) > 0:
    #         print(
    #             f"Удалены строки с более чем {int(threshold * 100)}% пропусков: {len(bad_rows)} шт."
    #         )
    #         df = df.drop(index=bad_rows)
    #     return df

    def _handle_nans(self, df):
        for col in self.numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        uvolen_series = df["уволен"] if "уволен" in df.columns else None

        df = self.drop_trash_feature(df)
        # df = self.drop_trash_rows(df)

        if uvolen_series is not None:
            uvolen_series = uvolen_series.loc[df.index]

        # Удаляем "уволен" ПЕРЕД определением признаков!
        if "уволен" in df.columns:
            df = df.drop(columns=["уволен"])

        # Категориальные
        self.cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        # for col in self.cat_cols:
        #     df[col] = df[col].astype(str).fillna("other")

        # Числовые
        self.numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        df = self._handle_nans(df)

        onehot_cols = [col for col in self.cat_cols if df[col].nunique() <= 15]
        ordinal_cols = [col for col in self.cat_cols if df[col].nunique() >= 16]

        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("onehot", self.onehot_encoder, onehot_cols),
                ("ordinal", self.ordinal_encoder, ordinal_cols),
                ("num", RobustScaler(), self.numeric_cols),
            ],
            remainder="drop",
        )

        df_transformed = self.preprocessor.fit_transform(df)
        df_transformed = pd.DataFrame(
            df_transformed, columns=self.preprocessor.get_feature_names_out()  # type: ignore
        )

        # Возвращаем 'уволен'
        if uvolen_series is not None:
            df_transformed["уволен"] = uvolen_series.values

        return df_transformed

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # Удаляем мусорные столбцы
        df.drop(columns=[col for col in DROP_COLS if col in df.columns], inplace=True)

        # Сохраняем целевую переменную
        uvolen_series = df["уволен"] if "уволен" in df.columns else None
        if uvolen_series is not None:
            uvolen_series = uvolen_series.loc[df.index]

        # Удаляем "уволен" перед препроцессингом
        if "уволен" in df.columns:
            df = df.drop(columns=["уволен"])

        # Обработка категориальных
        # for col in self.cat_cols:
        #     if col not in df.columns:
        #         df[col] = "other"
        #     df[col] = df[col].astype(str).fillna("other")

        df = self._handle_nans(df)

        # Применяем препроцессор
        df_transformed = self.preprocessor.transform(df)  # type: ignore
        df_transformed = pd.DataFrame(
            df_transformed, columns=self.preprocessor.get_feature_names_out()  # type: ignore
        )

        # Возвращаем "уволен"
        if uvolen_series is not None:
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


def merge_fillna(left_df, right_df, on="фио"):
    # Сливаем по ключу
    merged = pd.merge(left_df, right_df, on=on, how="outer", suffixes=("_left", "_right"))

    result = merged[[on]].copy()

    # Обрабатываем все колонки, кроме ключа
    for col in left_df.columns:
        if col == on:
            continue

        col_left = f"{col}_left"
        col_right = f"{col}_right"

        if col_left in merged.columns and col_right in merged.columns:
            # Если есть оба — заполняем пропуски значениями из правого
            result[col] = merged[col_left].combine_first(merged[col_right])
        elif col_left in merged.columns:
            result[col] = merged[col_left]
        elif col_right in merged.columns:
            result[col] = merged[col_right]

    return result


def convert_dates(df):
    try:
        date_columns = ["дата_рождения", "дата_увольнения", "дата_приема_в_1с"]

        # Замена пропусков на '1970-01-01 00:00:00'
        for col in date_columns:
            df[col] = df[col].fillna("1970-01-01 00:00:00")

        # Конвертация в datetime с учетом разных форматов
        df["дата_рождения"] = pd.to_datetime(
            df["дата_рождения"], format="%d.%m.%Y", errors="coerce"
        )
        df["дата_увольнения"] = pd.to_datetime(
            df["дата_увольнения"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
        )
        df["дата_приема_в_1с"] = pd.to_datetime(
            df["дата_приема_в_1с"], format="%Y-%m-%d", errors="coerce"
        )

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

        # --- Обработка детей ---
        children["date_birth"] = pd.to_datetime(
            children["date_birth"], format="%d.%m.%Y", errors="coerce"
        )
        children["age"] = ((datetime.now() - children["date_birth"]).dt.days / 365).round(1)

        grouped_children = (
            children.groupby("id")
            .agg(
                число_детей=("name", "count"),
                средний_возраст_детей=("age", "mean"),
                средний_пол_детей=("gender", mode_with_tie),
            )
            .reset_index()
        )

        # --- Приведение ФИО ---
        for df in [users_salary, users_cadr]:
            df["фио"] = df["фио"].str.split().str[:2].str.join(" ")
        main_users["фио"] = main_users["фамилия"] + " " + main_users["имя"]

        # --- Объединение с историей ---
        users_cadr = merge_fillna(users_cadr, history_cadr, on="фио")
        users_cadr.to_csv(
            INPUT_HISTORY_CADR, index=False, sep=",", decimal=",", encoding="utf-8-sig"
        )

        main_users = merge_base([main_users, users_cadr], "фио", "right")
        main_users = merge_base([main_users, users_salary], "фио", "left")
        main_users = merge_base([main_users, grouped_children], "id", "left")
        main_users = merge_base([main_users, director], "id", "left")

        # --- Очистка ---
        main_users = main_users[~main_users["логин"].isin(LOGINS_TO_REMOVE)]
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

        # --- Возраст и стаж ---
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

        # --- Кол-во подчинённых ---
        sub_count = main_users["id_руководителя"].value_counts()
        main_users["подчиненные"] = main_users["id"].apply(lambda x: sub_count.get(x, 0))

        # --- Преобразования ---
        main_users["id_руководителя"] = main_users["id_руководителя"].fillna(-1).astype(int)
        main_users["средний_возраст_детей"] = (
            main_users["средний_возраст_детей"].fillna(0).astype(float)
        )
        main_users["ср_зп"] = main_users["ср_зп"].fillna(0).astype(float)

        # --- Новые признаки ---
        main_users["скоро_др"] = (
            (main_users["дата_рождения"].dt.month == today.month)
            & (abs(main_users["дата_рождения"].dt.day - today.day) <= 30)
        ).astype(int)

        main_users["скоро_годовщика_приема"] = (
            (main_users["дата_приема_в_1с"].dt.month == today.month)
            & (abs(main_users["дата_приема_в_1с"].dt.day - today.day) <= 30)
        ).astype(int)

        main_users["есть_маленькие_дети"] = (
            main_users["id"].map(children.groupby("id")["age"].min()).le(5)
            & (main_users["число_детей"] > 0)
        ).astype(int)

        main_users["индес_ответственности_за_детьми"] = main_users["число_детей"].fillna(0) * (
            18 - main_users["средний_возраст_детей"].fillna(0)
        )

        mean_salary = main_users["ср_зп"].replace(0, np.nan).mean()
        main_users["зп_на_ср_зп_по_компании"] = main_users["ср_зп"] / mean_salary
        main_users["не_доплачивают"] = (main_users["зп_на_ср_зп_по_компании"] < 0.8).astype(int)

        main_users["стаж_на_возраст"] = main_users["стаж"] / main_users["возраст"]
        main_users["зп_на_число_детей"] = main_users["ср_зп"] / main_users["число_детей"].replace(
            0, np.nan
        )
        main_users["недоплата_и_больше_2_детей"] = (
            (main_users["не_доплачивают"] == 1) & (main_users["число_детей"] >= 2)
        ).astype(int)

        # --- Сохранение ---
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
    main_top = merge_base([stat, main_all], "логин", "right")
    main_top = main_top[main_top["логин"].isin(stat["логин"])]
    main_top = main_top[~main_top["логин"].isin(LOGINS_TO_REMOVE)]

    main_top["id_руководителя"] = main_top["id_руководителя"].fillna(-1).astype(int)
    main_top["средний_возраст_детей"] = main_top["средний_возраст_детей"].fillna(0).astype(float)
    main_top["ср_зп"] = main_top["ср_зп"].fillna(0).astype(float)

    for col in FLOAT_COLS:
        if col in main_top.columns:
            main_top[col] = pd.to_numeric(main_top[col], errors="coerce")
            main_top[col] = main_top[col].fillna(main_top[col].median())

    main_top.to_csv(f"{DATA_PROCESSED}/main_top.csv", index=False)

    # 👉 Новый препроцессор только для main_top
    preprocessor_top = DataPreprocessor()
    main_top_for_train = preprocessor_top.fit(main_top)

    main_top_for_train.to_csv(f"{DATA_PROCESSED}/main_top_for_train.csv", index=False)

    # 💾 Сохраним отдельный препроцессор
    preprocessor_top.save(f"{DATA_PROCESSED}/preprocessor_top")

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
