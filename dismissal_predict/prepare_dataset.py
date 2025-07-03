import os

import pandas as pd

INPUT_FILE_CADR = "data/raw/last_users_from_cadr.xls"
DATA_INTERIM = "data/interim"


def process_dataset(input_file, output_dir):
    # Чтение файла Excel, с указанием строки с заголовками если они не в первой
    df = pd.read_excel(input_file, header=2)

    print("Имена столбцов в файле:", df.columns.tolist())

    # Выбор нужных столбцов по именам
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
        print("Отсутствуют столбцы:", missing_columns)
        return

    processed_data = df[columns_to_copy]

    output_file = os.path.join(output_dir, "check_last_users_update.csv")
    processed_data.to_csv(output_file, index=False, encoding="utf-8")
    print("Файл успешно обработан и сохранен.")


if __name__ == "__main__":
    process_dataset(INPUT_FILE_CADR, DATA_INTERIM)
