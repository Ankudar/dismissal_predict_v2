import glob
import os
import re

import pandas as pd

INPUT_FILE_CADR = "data/raw/last_users_from_cadr.xls"
INPUT_FILE_CHILDREN = "data/raw/children.csv"
INPUT_FILE_MAIN_USERS = "data/raw/main_users.csv"
DATA_INTERIM = "data/interim"
INPUT_ZUP_PATH = "data/raw/zup"


def process_last_users_from_cadr(input_file, output_dir):
    df = pd.read_excel(input_file, header=2)
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

    output_file = os.path.join(output_dir, "check_last_users_update.csv")
    processed_data.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Файл {input_file} успешно обработан и сохранен.")


def process_children(input_file, output_dir):
    df = pd.read_csv(input_file, header=0)
    columns_to_copy = ["ID", "NAME", "DATE_BIRTH", "GENDER"]

    df.columns = df.columns.str.strip()
    missing_columns = [col for col in columns_to_copy if col not in df.columns]
    if missing_columns:
        return

    processed_data = df[columns_to_copy]

    output_file = os.path.join(output_dir, "children.csv")
    processed_data.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Файл {input_file} успешно обработан и сохранен.")


def process_main_users(input_file, output_dir):
    df = pd.read_csv(input_file, header=1)
    columns_to_copy = ["Дата увольнения", "Логин", "Должность", "Имя", "Фамилия"]

    df.columns = df.columns.str.strip()
    missing_columns = [col for col in columns_to_copy if col not in df.columns]
    if missing_columns:
        return

    processed_data = df[columns_to_copy]

    output_file = os.path.join(output_dir, "main_users.csv")
    processed_data.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Файл {input_file} успешно обработан и сохранен.")


def process_zup_path(input_dir, output_dir):
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
            print(f"Ошибка: недостаточно столбцов в {filename}")
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
        merged_data[col] = pd.to_numeric(merged_data[col], errors="coerce").fillna(0).astype(float)

    date_columns = sorted(
        merged_data.columns[1:],
        key=lambda x: pd.to_datetime(f"{x.split('.')[1]}-{x.split('.')[0]}-01"),
    )
    sorted_columns = ["Месяц"] + date_columns
    merged_data = merged_data[sorted_columns]

    # Создаем новый DataFrame с ФИО и средним значением
    result_data = pd.DataFrame()
    result_data["ФИО"] = merged_data["Месяц"]
    result_data["ср_зп"] = round(merged_data.iloc[:, 1:].mean(axis=1), 2)

    # Фильтрация ФИО
    unique_fios = result_data["ФИО"].unique()
    pattern = re.compile(r"^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+")
    filtered_fios = [fio for fio in unique_fios if pattern.match(fio)]
    result_data_filtered = result_data[result_data["ФИО"].isin(filtered_fios)]
    result_data_filtered = result_data_filtered.reset_index(drop=True)

    # Градация значений
    num_categories = 5
    result_data_filtered["уровень_зп"] = pd.qcut(
        result_data_filtered["ср_зп"],
        q=num_categories,
        labels=["низкий", "ниже среднего", "средний", "выше среднего", "высокий"],
    )

    # Сохраняем итоговые данные в CSV
    result_file = os.path.join(output_dir, "zup.csv")
    result_data_filtered.to_csv(result_file, index=False)


if __name__ == "__main__":
    process_last_users_from_cadr(INPUT_FILE_CADR, DATA_INTERIM)
    process_children(INPUT_FILE_CHILDREN, DATA_INTERIM)
    process_main_users(INPUT_FILE_MAIN_USERS, DATA_INTERIM)
    process_zup_path(INPUT_ZUP_PATH, DATA_INTERIM)
