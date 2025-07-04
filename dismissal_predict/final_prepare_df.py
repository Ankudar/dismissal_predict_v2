from datetime import datetime

import numpy as np
import pandas as pd

DATA_INTERIM = "data/interim"
DATA_PROCESSED = "data/processed"
INPUT_FILE_MAIN_USERS = f"{DATA_INTERIM}/main_users.csv"
INPUT_FILE_CADR = f"{DATA_INTERIM}/check_last_users_update.csv"
INPUT_FILE_CHILDREN = f"{DATA_INTERIM}/children.csv"
INPUT_FILE_STAT = f"{DATA_INTERIM}/whisper_stat.csv"
INPUT_ZUP_PATH = f"{DATA_INTERIM}/zup.csv"

main_users = pd.read_csv(INPUT_FILE_MAIN_USERS, delimiter=",", decimal=",")
users_cadr = pd.read_csv(INPUT_FILE_CADR, delimiter=",", decimal=",")
users_salary = pd.read_csv(INPUT_ZUP_PATH, delimiter=",", decimal=",")
children = pd.read_csv(INPUT_FILE_CHILDREN, delimiter=",", decimal=",")
stat = pd.read_csv(INPUT_FILE_STAT, delimiter=",", decimal=",")


def merge_base(bases, index, merge_type):
    merged_df = bases[0]
    for base in bases[1:]:
        merged_df = pd.merge(merged_df, base, on=index, how=merge_type)

    return merged_df


def convert_dates(df):
    target_format = "%d.%m.%Y %H:%M:%S"

    df["дата_рождения"] = df["дата_рождения"].apply(
        lambda x: (
            pd.to_datetime(x, format="%d.%m.%Y", errors="coerce").strftime(target_format)
            if pd.notna(x)
            else x
        )
    )

    df["дата_приема_в_1с"] = df["дата_приема_в_1с"].apply(
        lambda x: (
            pd.to_datetime(x, format="%Y-%m-%d", errors="coerce").strftime(target_format)
            if pd.notna(x)
            else x
        )
    )

    date_columns = ["дата_увольнения", "дата_рождения", "дата_приема_в_1с"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format=target_format, errors="coerce")

    return df


def mode_with_tie(series):
    if series.empty:
        return "none"

    counts = series.value_counts()
    max_count = counts.max()
    modes = counts[counts == max_count].index.tolist()

    if "m/f" in modes:
        modes = ["m/f" if mode == "f/m" else mode for mode in modes]
    return "/".join(set(modes))


def main_prepare_for_all(main_users, users_salary, users_cadr, children):
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
            main_child_gender=("gender", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
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
    main_users["грейд"] = pd.to_numeric(
        main_users["грейд"].astype(str).str.extract("(\d+)")[0], errors="coerce"
    )
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
    main_top.to_csv(f"{DATA_PROCESSED}/main_top.csv", index=False)
    pass


if __name__ == "__main__":
    main_prepare_for_all(main_users, users_salary, users_cadr, children)
    prepare_with_mic()
