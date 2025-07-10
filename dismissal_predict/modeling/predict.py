from datetime import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PROCESSED = "data/processed"
MODELS = "models"
RESULTS = "data/results"
os.makedirs(RESULTS, exist_ok=True)

INPUT_FILE_MAIN_ALL = f"{DATA_PROCESSED}/main_all.csv"
INPUT_FILE_MAIN_TOP = f"{DATA_PROCESSED}/main_top.csv"

MODEL_MAIN_USERS = f"{MODELS}/xgb_main_users.pkl"
MODEL_TOP_USERS = f"{MODELS}/xgb_top_users.pkl"

RESULT_FILE_ALL = f"{RESULTS}/result_all.xlsx"
RESULT_FILE_TOP = f"{RESULTS}/result_top.xlsx"

main_all = pd.read_csv(INPUT_FILE_MAIN_ALL, delimiter=",", decimal=",")
main_top = pd.read_csv(INPUT_FILE_MAIN_TOP, delimiter=",", decimal=",")

model_main_users = joblib.load(MODEL_MAIN_USERS)
model_top_users = joblib.load(MODEL_TOP_USERS)


def clean_encode_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Отделяем таргет
    target_col = "уволен"
    if target_col in df.columns:
        target = df[[target_col]].values.flatten()
        df = df.drop(columns=[target_col])
    else:
        target = None

    # Удаляем ненужные столбцы
    drop_cols = [
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
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    # Преобразование категориальных признаков
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Удаляем datetime-признаки, если остались
    df = df.drop(columns=df.select_dtypes(include=["datetime"]).columns)

    # Масштабирование числовых признаков
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if not df[numeric_cols].empty:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Возвращаем таргет
    if target is not None:
        df[target_col] = target

    return df


def add_predictions_to_excel(data, model, result_file, threshold):
    # Фильтрация данных для предсказания
    filtered_data = data[data["дата_увольнения"].isna()]

    # Очистка и предобработка данных
    cleaned_data = clean_encode_scale(filtered_data)

    # Предсказания модели
    probabilities = model.predict_proba(cleaned_data.drop(columns=["уволен"]))[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Создание DataFrame с результатами
    results = pd.DataFrame(
        {
            "фио": filtered_data["фио"].values,
            "уволен": 0,
            "предсказание_увольнения": predictions,
        }
    )

    # Обновление столбца "уволен" для тех, у кого есть дата увольнения
    data["уволен"] = data["дата_увольнения"].notna().astype(int)

    # Проверка наличия файла и чтение существующих данных
    if os.path.exists(result_file):
        existing_results = pd.read_excel(result_file)
    else:
        existing_results = pd.DataFrame()

    # Текущая дата
    current_date = datetime.now().strftime("%d.%m.%Y")

    # Добавление столбца с текущей датой, если его нет
    if current_date not in existing_results.columns:
        existing_results[current_date] = None

    # Проверка наличия столбца 'фио' в существующих результатах
    if "фио" not in existing_results.columns:
        existing_results["фио"] = None

    # Обновление существующих данных новыми предсказаниями
    for index, row in results.iterrows():
        fio = row["фио"]
        if fio in existing_results["фио"].values:
            existing_results.loc[existing_results["фио"] == fio, current_date] = probabilities[
                index
            ]
        else:
            new_row = row.to_dict()
            new_row[current_date] = probabilities[index]
            existing_results = existing_results.append(new_row, ignore_index=True)

    # Запись результатов в Excel
    existing_results.to_excel(result_file, index=False)


def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(1.0, 0.0, -0.001)
    costs = []

    for threshold in thresholds:
        predictions = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Определяем "стоимость" на основе ваших требований
        cost = (1 * tn) - (1 * fp) - (1 * fn) + (1 * tp)
        costs.append(cost)

    # Находим порог, при котором "стоимость" максимальна
    optimal_threshold_index = np.argmax(costs)
    optimal_threshold = thresholds[optimal_threshold_index]

    return optimal_threshold


def print_confusion_matrix(data, model):
    # Очистка и предобработка данных
    cleaned_data = clean_encode_scale(data)

    # Предсказания модели
    probabilities = model.predict_proba(cleaned_data.drop(columns=["уволен"]))[:, 1]

    # Фактические значения
    actual = cleaned_data["уволен"].astype(float).astype(int)  # Преобразуем в int

    # Подбор лучшего порога
    best_threshold = find_best_threshold(actual, probabilities)

    # Применение лучшего порога
    predictions = (probabilities >= best_threshold).astype(int)

    # Матрица ошибок
    cm = confusion_matrix(actual, predictions)
    print("Confusion Matrix with best threshold:")
    print(cm)

    # Отчет о классификации
    report = classification_report(actual, predictions)
    print("\nClassification Report:")
    print(report)

    print(f"\nBest threshold: {best_threshold}")

    return best_threshold


if __name__ == "__main__":
    print("Confusion Matrix and Classification Report for main_top:")
    best_threshold_top = print_confusion_matrix(main_top, model_top_users)
    add_predictions_to_excel(main_top, model_top_users, RESULT_FILE_TOP, best_threshold_top)

    print("\nConfusion Matrix and Classification Report for main_all:")
    best_threshold_all = print_confusion_matrix(main_all, model_main_users)
    add_predictions_to_excel(main_all, model_main_users, RESULT_FILE_ALL, best_threshold_all)
