from datetime import datetime
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rusgenderdetection import get_gender  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    "число_детей",
    "не_доплачивают",
    "зп_на_число_детей",
    "id_руководителя",
    "зп_на_ср_зп_по_компании",
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

    def drop_trash_feature(self, df, threshold=0.8):
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

        if uvolen_series is not None:
            uvolen_series = uvolen_series.loc[df.index]

        if "уволен" in df.columns:
            df = df.drop(columns=["уволен"])

        # Попробуем привести потенциальные числа к числовому типу
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        # Обновим числовые и категориальные признаки после приведения типов
        self.numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df = self._handle_nans(df)

        onehot_cols = [col for col in self.cat_cols if df[col].nunique() <= 10]
        ordinal_cols = [col for col in self.cat_cols if df[col].nunique() > 10]

        self.onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        )
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
            df_transformed, columns=self.preprocessor.get_feature_names_out() # type: ignore
        )

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
    merged = pd.merge(left_df, right_df, on=on, how="outer", suffixes=("_left", "_right"))

    result = merged[[on]].copy()

    for col in left_df.columns:
        if col == on:
            continue

        col_left = f"{col}_left"
        col_right = f"{col}_right"

        if col_left in merged.columns and col_right in merged.columns:
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


def extract_first_name(full_name):
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
                число_детей=("name", "count"),
                средний_возраст_детей=("age", "mean"),
                дети_мальчики=("gender", lambda x: (x == "m").sum()),
                дети_девочки=("gender", lambda x: (x == "f").sum()),
            )
            .reset_index()
        )

        for df in [users_salary, users_cadr]:
            df["фио"] = df["фио"].str.split().str[:2].str.join(" ")
        main_users["фио"] = main_users["фамилия"] + " " + main_users["имя"]

        users_cadr = merge_fillna(users_cadr, history_cadr, on="фио")
        users_cadr.to_csv(
            INPUT_HISTORY_CADR, index=False, sep=",", decimal=",", encoding="utf-8-sig"
        )

        main_users = merge_base([main_users, users_cadr], "фио", "right")
        main_users = merge_base([main_users, users_salary], "фио", "left")
        main_users = merge_base([main_users, grouped_children], "id", "left")

        for col in ["число_детей", "дети_мальчики", "дети_девочки"]:
            if col in main_users.columns:
                main_users[col] = main_users[col].fillna(0).round().astype(int)

        main_users = merge_base([main_users, director], "id", "left")

        main_users = main_users[~main_users["логин"].isin(LOGINS_TO_REMOVE)]
        main_users["пол"] = main_users["фио"].apply(determine_gender)
        main_users.replace("nan", pd.NA, inplace=True)
        main_users = main_users.dropna(subset=["имя", "фамилия"])
        main_users = convert_dates(main_users)

        for col in ["логин", "должность", "имя", "фамилия"]:
            main_users[col] = main_users[col].astype(str)
        main_users = main_users[main_users["должность"] != "уборщица"]

        main_users["ср_зп"] = main_users["ср_зп"].fillna(0)
        main_users["ср_зп"] = main_users["ср_зп"].astype(float)
        main_users = main_users[main_users["ср_зп"] >= 15000]

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

        sub_count = main_users["id_руководителя"].value_counts()
        main_users["подчиненные"] = main_users["id"].apply(lambda x: sub_count.get(x, 0))

        main_users["id_руководителя"] = main_users["id_руководителя"].fillna(-1).astype(str)
        main_users["средний_возраст_детей"] = (
            main_users["средний_возраст_детей"].fillna(0).astype(float)
        )

        main_users["скоро_др"] = (
            (main_users["дата_рождения"].dt.month == today.month)
            & (abs(main_users["дата_рождения"].dt.day - today.day) <= 30)
        ).astype(int)

        main_users["скоро_годовщина_приема"] = (
            (main_users["дата_приема_в_1с"].dt.month == today.month)
            & ((main_users["дата_приема_в_1с"].dt.day - today.day).abs() <= 30)
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

        main_users["зп_на_число_детей"] = np.where(
            main_users["число_детей"] == 0,
            main_users["ср_зп"],
            main_users["ср_зп"] / main_users["число_детей"],
        )

        main_users["недоплата_и_больше_2_детей"] = (
            (main_users["не_доплачивают"] == 1) & (main_users["число_детей"] >= 2)
        ).astype(int)

        main_users["зп_к_возрасту"] = main_users["ср_зп"] / (main_users["возраст"] + 1)
        main_users["зп_к_стажу"] = main_users["ср_зп"] / (main_users["стаж"] + 1)
        # main_users["стаж_к_возрасту"] = main_users["стаж"] / (main_users["возраст"] + 1)

        main_all_path = f"{DATA_PROCESSED}/main_all.csv"
        main_users_updated = update_existing_data(main_users, main_all_path, id_col="id")
        main_users_updated.to_csv(main_all_path, index=False)
        main_users_updated.to_csv(
            f"{DATA_PROCESSED}/main_all_history_do_not_tuch.csv", index=False
        )

        preprocessor = DataPreprocessor()
        main_users_for_train = preprocessor.fit(main_users_updated)

        # Проверка и удаление строк с NaN
        nan_rows = main_users_for_train.isnull().any(axis=1)
        n_removed = nan_rows.sum()

        if n_removed > 0:
            print("Обнаружены NaN в строках — удаляем:")
            print(main_users_for_train.isnull().sum()[main_users_for_train.isnull().any()])
            main_users_for_train = main_users_for_train[~nan_rows]
            print(f"Удалено строк: {n_removed}")
        else:
            print("NaN не обнаружены в main_all_for_train — удаление не требуется.")

        # Сохраняем финальные данные и препроцессор
        main_users_for_train.to_csv(f"{DATA_PROCESSED}/main_users_for_train.csv", index=False)
        preprocessor.save(f"{DATA_PROCESSED}/preprocessor.pkl")

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def prepare_with_mic():
    main_all = pd.read_csv(f"{DATA_PROCESSED}/main_all.csv", delimiter=",", decimal=",")
    main_top = merge_base([stat, main_all], "логин", "right")
    main_top = main_top[main_top["логин"].isin(stat["логин"])]
    main_top = main_top[~main_top["логин"].isin(LOGINS_TO_REMOVE)]

    main_top["средний_возраст_детей"] = main_top["средний_возраст_детей"].fillna(0).astype(float)

    for col in FLOAT_COLS:
        if col in main_top.columns:
            main_top[col] = pd.to_numeric(main_top[col], errors="coerce")
            main_top[col] = main_top[col].fillna(main_top[col].median())

    main_top_path = f"{DATA_PROCESSED}/main_top.csv"
    main_top_updated = update_existing_data(main_top, main_top_path, id_col="id")
    main_top_updated.to_csv(main_top_path, index=False)
    main_top_updated.to_csv(f"{DATA_PROCESSED}/main_top_history_do_not_tuch.csv", index=False)

    preprocessor_top = DataPreprocessor()
    main_top_for_train = preprocessor_top.fit(main_top_updated)

    # Проверка и удаление строк с NaN
    nan_rows = main_top_for_train.isnull().any(axis=1)
    n_removed = nan_rows.sum()

    if n_removed > 0:
        print("Обнаружены NaN в строках main_top_for_train — удаляем:")
        print(main_top_for_train.isnull().sum()[main_top_for_train.isnull().any()])
        main_top_for_train = main_top_for_train[~nan_rows]
        print(f"Удалено строк: {n_removed}")
    else:
        print("NaN не обнаружены в main_top_for_train — удаление не требуется.")

    # Сохраняем финальные данные и препроцессор
    main_top_for_train.to_csv(f"{DATA_PROCESSED}/main_top_for_train.csv", index=False)
    preprocessor_top.save(f"{DATA_PROCESSED}/preprocessor_top.pkl")


def calc_target_correlations(df, target_col: str = "уволен", file_path: str = "data.csv"):
    """
    Считает корреляции признаков с таргетом, строит heatmap и рассчитывает VIF.
    Все результаты сохраняются в виде файлов рядом с file_path.
    """
    folder = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    df_tmp = df.copy()

    cat_cols = df_tmp.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        df_tmp[c] = df_tmp[c].astype("category").cat.codes

    numeric_cols = df_tmp.select_dtypes(exclude=["object", "category"]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"target_col должен быть числовым")

    corr_df = (
        df_tmp[numeric_cols]
        .corr()[target_col]
        .drop(target_col)
        .sort_values(key=np.abs, ascending=False)
    )
    pearson_path = os.path.join(folder, f"{base}_pearson_target_corr.csv")
    corr_df.to_csv(pearson_path)

    # Исключаем DROP_COLS, но оставляем target_col
    heatmap_cols = [col for col in numeric_cols if col not in DROP_COLS or col == target_col]
    corr_matrix = df_tmp[heatmap_cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, interpolation="nearest", cmap="coolwarm", aspect="auto")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=8)
    plt.colorbar()
    plt.title("Correlation Heatmap (включая target)")

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            value = corr_matrix.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")

    heatmap_path = os.path.join(folder, f"{base}_features_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # --- VIF ---
    vif_cols = [col for col in numeric_cols if col != target_col and col not in DROP_COLS]
    X_vif = df_tmp[vif_cols].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_vif), columns=vif_cols)

    vif_data = pd.DataFrame()
    vif_data["feature"] = vif_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    vif_path = os.path.join(folder, f"{base}_vif.csv")
    vif_data.to_csv(vif_path, index=False)

    print(f"Target corr saved to:   {pearson_path}")
    print(f"Heatmap saved to:       {heatmap_path}")
    print(f"VIF table saved to:     {vif_path}")

    return corr_df, vif_data


def update_existing_data(
    new_df: pd.DataFrame, existing_path: str, id_col: str = "id"
) -> pd.DataFrame:
    if os.path.exists(existing_path):
        old_df = pd.read_csv(existing_path, delimiter=",", decimal=",")

        # Явное приведение id к строке или int (рекомендуется int, если id — числовой)
        old_df[id_col] = old_df[id_col].astype(str)
        new_df[id_col] = new_df[id_col].astype(str)

        # Найдём общие колонки
        common_cols = sorted(list(set(old_df.columns).intersection(set(new_df.columns))))

        old_df = old_df[common_cols].copy()
        new_df = new_df[common_cols].copy()

        old_df = old_df.drop_duplicates(subset=[id_col])
        new_df = new_df.drop_duplicates(subset=[id_col])

        old_df.set_index(id_col, inplace=True)
        new_df.set_index(id_col, inplace=True)

        old_df = old_df.sort_index().sort_index(axis=1)
        new_df = new_df.sort_index().sort_index(axis=1)

        # Сравнение
        common_index = old_df.index.intersection(new_df.index)
        old_common = old_df.loc[common_index]
        new_common = new_df.loc[common_index]

        comparison = old_common.fillna("NAN") != new_common.fillna("NAN")
        changed_mask = comparison.any(axis=1)

        updated_df = old_df.copy()
        updated_df.update(new_common[changed_mask])

        new_ids = new_df.index.difference(old_df.index)
        if not new_ids.empty:
            updated_df = pd.concat([updated_df, new_df.loc[new_ids]])

        updated_df = updated_df.reset_index()
    else:
        updated_df = new_df.copy()

    return updated_df


def run_all():
    main_prepare_for_all(main_users, users_salary, users_cadr, children)
    prepare_with_mic()


if __name__ == "__main__":
    logger.info("Финальная подготовка баз началась")
    run_all()
    main_all_history_do_not_tuch = pd.read_csv(
        f"{DATA_PROCESSED}/main_all_history_do_not_tuch.csv", delimiter=",", decimal=","
    )
    corr = calc_target_correlations(
        main_all_history_do_not_tuch,
        target_col="уволен",
        file_path=f"{DATA_PROCESSED}/main_all_history_do_not_tuch.csv",
    )

    logger.info("Финальная подготовка баз завершилась")
