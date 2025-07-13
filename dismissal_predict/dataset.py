from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import shutil
from threading import Lock
import time
import traceback

from config import MAIN_CONFIGS, Config
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm

DATA_RAW = "~/data/raw"
CHROMEDRIVER_PATH = "/home/root6/python/CHROMEDRIVER/chromedriver"
CONFIG = Config(MAIN_CONFIGS)
lock = Lock()

login_1c, password_1c, Srvr_zup, Ref_zup = CONFIG.get_1c_info()
cadr_users_list_url = CONFIG.get_cadr_users_list()
portal_children_link = CONFIG.get_portal_children_link()
portal_login, portal_password = CONFIG.get_portal_credentials()
portal_users_link = CONFIG.get_portal_users_link()
whisper_data = CONFIG.get_whisper_url()
WHISPER_CATEGORIES = CONFIG.WHISPER_CATEGORIES


def is_valid_date(filename):
    try:
        datetime.strptime(filename, "%d.%m.%Y.xls")
        return True
    except ValueError:
        return False


def get_latest_file(directory):
    latest_file = None
    latest_date = None

    for root, _, files in os.walk(directory):
        for file in files:
            if is_valid_date(file):
                file_date = datetime.strptime(file, "%d.%m.%Y.xls")
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = os.path.join(root, file)

    if latest_file:
        destination_file = os.path.join(DATA_RAW, "last_users_from_cadr.xls")
        shutil.copy(latest_file, destination_file)
    else:
        print("Файлы не найдены.")


def login(driver, username, password):
    login_input = driver.find_element(
        By.CSS_SELECTOR, "#authorize > div > div:nth-child(3) > div.login-input-wrap > input"
    )
    login_input.send_keys(username)

    password_input = driver.find_element(
        By.CSS_SELECTOR, "#authorize_password > div.login-input-wrap > input"
    )
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)
    time.sleep(2)


def get_driver():
    driver_service = Service(executable_path=CHROMEDRIVER_PATH)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    # Инициализация драйвера
    driver = webdriver.Chrome(service=driver_service, options=options)
    return driver


def get_portal_users(auth_url):
    # auth_url = "http://next.portal.local/bitrix/admin/user_admin.php?lang=ru#authorize"
    driver = None
    try:
        driver = get_driver()
        driver.get(auth_url)
        login(driver, portal_login, portal_password)

        driver.get(auth_url)
        downloads_path = os.path.expanduser("~/Downloads")
        time.sleep(5)

        files = os.listdir(downloads_path)
        files = [f for f in files if f.endswith(".xls") or f.endswith(".xlsx")]

        if not files:
            print("Нет загруженных файлов.")
            return

        latest_file = max([os.path.join(downloads_path, f) for f in files], key=os.path.getctime)

        temp_file_path = os.path.join(DATA_RAW, "main_users.html")
        shutil.copy(latest_file, temp_file_path)

        df_list = pd.read_html(temp_file_path)
        if df_list:
            df = df_list[0]
            new_file_path = os.path.join(DATA_RAW, "main_users.csv")
            df.to_csv(new_file_path, index=False)

            print(f"Файл {latest_file} скопирован в {DATA_RAW} и пересохранен как main_users.csv.")
        else:
            print("Не удалось найти таблицы в HTML-файле.")

        os.remove(latest_file)
        os.remove(temp_file_path)

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if driver:
            driver.quit()


def get_portal_children(auth_url):
    driver = None
    try:
        driver = get_driver()
        driver.get(auth_url)
        login(driver, portal_login, portal_password)

        table = driver.find_element(
            By.CSS_SELECTOR, "#form_tbl_perfmon_table6f366e427324b863457a9faef450aed6"
        )
        rows = table.find_elements(By.TAG_NAME, "tr")

        data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            data.append([col.text for col in cols])

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(DATA_RAW, "children.csv"), index=False, header=False, sep=",")
        print(f"Данные сохранены в {DATA_RAW}/children.csv.")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if driver:
            driver.quit()


def get_whisper_stat(
    base_dir,
    check_list_file=f"{DATA_RAW}/check_list.txt",
    output_file=f"{DATA_RAW}/whisper_stat.csv",
):
    if not os.path.exists(check_list_file):
        with open(check_list_file, "w", encoding="utf-8") as f:
            f.write("")

    with open(check_list_file, "r", encoding="utf-8") as f:
        check_list = set(line.strip() for line in f)

    all_files = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                all_files.add(full_path)

    files_to_process = all_files - check_list

    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                process_file, file, check_list_file, output_file, files_to_process
            ): file
            for file in files_to_process
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files", unit="file"
        ):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                print(traceback.format_exc())

    print(f"Всего {len(all_files)} файлов.")
    print(f"Было обработано ранее {len(check_list)} файлов.")
    print(f"Было обработано сейчас {len(files_to_process)} файлов.")


def process_file(file, check_list_file, output_file, files_to_process):
    with open(check_list_file, "a", encoding="utf-8") as f:
        f.write(file + "\n")

    try:
        summ_check = 0
        dialog_analysis = {key: "нет" for key in WHISPER_CATEGORIES.keys()}
        dialog_analysis["Тон"] = "дружественный"  # По умолчанию

        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 8:
                return
            lines = lines[:8]  # Берем первые строки
            for line in lines:
                line = line.lower().strip()
                if line.startswith("[увольнение]"):
                    dialog_analysis["Увольнение"] = "да" if "да" in line else "нет"
                elif line.startswith("[оффер]"):
                    dialog_analysis["Оффер"] = "да" if "да" in line else "нет"
                elif line.startswith("[вредительство]"):
                    dialog_analysis["Вредительство"] = "да" if "да" in line else "нет"
                elif line.startswith("[конфликты]"):
                    dialog_analysis["Конфликты"] = "да" if "да" in line else "нет"
                elif line.startswith("[стресс]"):
                    dialog_analysis["Стресс"] = "да" if "да" in line else "нет"
                elif line.startswith("[личная жизнь]"):
                    dialog_analysis["Личная жизнь"] = "да" if "да" in line else "нет"
                elif line.startswith("[тон]"):
                    dialog_analysis["Тон"] = (
                        "негативный" if "негативный" in line else "дружественный"
                    )

        for category, value in dialog_analysis.items():
            if value == "да":
                base_weight = WHISPER_CATEGORIES[category]["да"]
                summ_check += base_weight

        if summ_check == 0:
            return

        parts = file.split("-")
        username = parts[2].split("@")[0]

        new_row = {
            "логин": username,
            "тон": WHISPER_CATEGORIES["Тон"][dialog_analysis["Тон"]],
            "увольнение": WHISPER_CATEGORIES["Увольнение"][dialog_analysis["Увольнение"]],
            "оффер": WHISPER_CATEGORIES["Оффер"][dialog_analysis["Оффер"]],
            "вредительство": WHISPER_CATEGORIES["Вредительство"][dialog_analysis["Вредительство"]],
            "личная жизнь": WHISPER_CATEGORIES["Личная жизнь"][dialog_analysis["Личная жизнь"]],
            "стресс": WHISPER_CATEGORIES["Стресс"][dialog_analysis["Стресс"]],
            "конфликты": WHISPER_CATEGORIES["Конфликты"][dialog_analysis["Конфликты"]],
        }

        with lock:
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                df = pd.DataFrame([new_row])
            else:
                df = pd.read_csv(output_file)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df.to_csv(output_file, encoding="utf-8", index=False)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        print(traceback.format_exc())


def get_1c_zup(base_name, server_1c, login, password):
    print("Типа выгрузил")


if __name__ == "__main__":
    get_latest_file(cadr_users_list_url)
    get_portal_users(portal_users_link)
    get_portal_children(portal_children_link)
    get_whisper_stat(whisper_data)
    get_1c_zup(Ref_zup, Srvr_zup, login_1c, password_1c)
