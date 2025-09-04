import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
from difflib import get_close_matches
import logging
import os
import shutil
from threading import Lock
import time

from config import MAIN_CONFIGS, Config

# import getpasspip
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_RAW = "/home/root6/python/dismissal_predict_v2/data/raw"
CHROMEDRIVER_PATH = "/home/root6/chromedriver/chromedriver"
CONFIG = Config(MAIN_CONFIGS)
lock = Lock()

login_1c, password_1c, Srvr_zup, Ref_zup = CONFIG.get_1c_info()
cadr_users_list_url = CONFIG.get_cadr_users_list()
portal_children_link = CONFIG.get_portal_children_link()
portal_login, portal_password = CONFIG.get_portal_credentials()
portal_users_link = CONFIG.get_portal_users_link()
portal_director_link = CONFIG.get_portal_director_link()
whisper_data = CONFIG.get_whisper_url()
WHISPER_CATEGORIES = CONFIG.WHISPER_CATEGORIES


def is_valid_date(filename):
    try:
        datetime.strptime(filename, "%d.%m.%Y.xls")
        return True
    except ValueError:
        return False


def get_latest_file(directory):
    logger.info("📥 Загрузка кадровых файлов")
    latest_file = None
    latest_date = None

    for root, _, files in os.walk(directory):
        for file in files:
            if is_valid_date(file):
                try:
                    file_date = datetime.strptime(file, "%d.%m.%Y.xls")
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = os.path.join(root, file)
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при разборе даты файла {file}: {e}")

    if latest_file:
        destination_file = os.path.join(DATA_RAW, "last_users_from_cadr.xls")
        shutil.copy(latest_file, destination_file)
        logger.info(f"✅ Скопирован файл: {latest_file}")
    else:
        logger.warning("❌ Файлы не найдены.")

    logger.info("📦 Загрузка кадровых файлов завершена")


def login(driver, username, password, auth_url=None):
    # Настройки времени ожидания
    WAIT_TIMEOUT = 30  # секунд

    # Выбор селекторов в зависимости от URL
    if auth_url and ("user_admin" in auth_url or "i_employees_children" in auth_url):
        # Страница с Bitrix user_admin
        login_selector = "#authorize > div > div:nth-child(3) > div.login-input-wrap > input"
        password_selector = "#authorize_password > div.login-input-wrap > input"
    else:
        # Обычная форма авторизации
        login_selector = (
            "#workarea-content > div > div > form > div:nth-child(4) > div:nth-child(1) > input"
        )
        password_selector = (
            "#workarea-content > div > div > form > div:nth-child(4) > div:nth-child(2) > input"
        )

    try:
        # Ожидание появления поля логина
        login_input = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, login_selector))
        )
        login_input.clear()
        login_input.send_keys(username)

        # Ожидание появления поля пароля
        password_input = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, password_selector))
        )
        password_input.clear()
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)

        # Ожидание завершения авторизации (можно настроить под ожидание конкретного элемента после входа)
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Короткая пауза для стабилизации
        time.sleep(2)

        logger.info("Авторизация прошла успешно")

    except TimeoutException:
        logger.error("Таймаут при ожидании элементов авторизации")
        raise
    except Exception as e:
        logger.error(f"Ошибка при авторизации: {e}")
        raise


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
    logger.info("Загрузка пользователей с портала")

    # Настройки времени ожидания (в секундах)
    PAGE_LOAD_TIMEOUT = 60  # время ожидания загрузки страницы
    IMPLICIT_WAIT = 30  # неявное ожидание элементов
    EXPLICIT_WAIT = 60  # явное ожидание для конкретных элементов
    DOWNLOAD_WAIT = 120  # время ожидания скачивания файла

    driver = None
    try:
        driver = get_driver()

        # Установка таймаутов
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        driver.implicitly_wait(IMPLICIT_WAIT)

        # Загрузка страницы авторизации
        driver.get(auth_url)

        # Авторизация
        login(driver, portal_login, portal_password, auth_url=auth_url)

        # Повторная загрузка страницы после авторизации
        driver.get(auth_url)

        # Явное ожидание загрузки страницы (можно настроить под конкретные элементы)
        try:
            WebDriverWait(driver, EXPLICIT_WAIT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception as e:
            logger.warning(f"Таймаут ожидания загрузки страницы: {e}")

        # Ожидание скачивания файла
        downloads_path = os.path.expanduser("~/Downloads")

        # Ждем появления файла в течение DOWNLOAD_WAIT секунд
        start_time = time.time()
        file_found = False

        while time.time() - start_time < DOWNLOAD_WAIT:
            files = os.listdir(downloads_path)
            xls_files = [f for f in files if f.endswith((".xls", ".xlsx"))]

            if xls_files:
                file_found = True
                break

            time.sleep(5)  # проверяем каждые 5 секунд

        if not file_found:
            logger.error("Файл не был загружен в течение отведенного времени")
            return

        # Находим самый свежий файл
        xls_files = [f for f in os.listdir(downloads_path) if f.endswith((".xls", ".xlsx"))]
        latest_file = max(
            [os.path.join(downloads_path, f) for f in xls_files], key=os.path.getctime
        )

        # Дополнительная проверка, что файл полностью загружен
        file_size = -1
        while True:
            current_size = os.path.getsize(latest_file)
            if current_size == file_size:
                break  # размер файла стабилизировался
            file_size = current_size
            time.sleep(2)

        # Обработка файла
        temp_file_path = os.path.join(DATA_RAW, "main_users.html")
        shutil.copy(latest_file, temp_file_path)

        # Чтение и сохранение данных
        df_list = pd.read_html(temp_file_path)
        if df_list:
            df = df_list[0]
            new_file_path = os.path.join(DATA_RAW, "main_users.csv")
            df.to_csv(new_file_path, index=False)
            logger.info(
                f"Файл {latest_file} скопирован в {DATA_RAW} и пересохранен как main_users.csv."
            )
        else:
            logger.error("Не удалось найти таблицы в HTML-файле.")
            return

        # Очистка временных файлов
        try:
            os.remove(latest_file)
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Ошибка при удалении временных файлов: {e}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise
    finally:
        if driver:
            driver.quit()

    logger.info("Загрузка пользователей с портала завершена")


def get_portal_children(auth_url):
    logger.info("Загрузка детей пользователей с портала")
    driver = None
    try:
        driver = get_driver()
        driver.get(auth_url)
        time.sleep(5)
        login(driver, portal_login, portal_password, auth_url=auth_url)
        time.sleep(5)

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
        logger.info(f"Данные сохранены в {DATA_RAW}/children.csv.")

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise
    finally:
        if driver:
            driver.quit()
    logger.info("Загрузка детей пользователей с портала завершена")


def get_portal_director(auth_url: str):
    logger.info("Загрузка руководителей пользователей с портала")
    driver = None

    try:
        driver = get_driver()
        driver.get(auth_url)
        login(driver, portal_login, portal_password)

        # Поиск таблицы
        table = driver.find_element(By.CSS_SELECTOR, "#workarea-content > div > table")
        rows = table.find_elements(By.TAG_NAME, "tr")

        data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if cols:
                data.append([col.text.strip() for col in cols])

        if not data:
            logger.warning("Таблица пуста или не найдены строки с данными.")
            return

        # Названия столбцов
        columns = ["id", "фио", "id_руководителя", "фио_руководителя"]
        df_new = pd.DataFrame(data, columns=columns[: len(data[0])])

        # Приведение ID к int с подстановкой значений по умолчанию
        def clean_id(val, default):
            try:
                return int(val)
            except:
                return default

        df_new["id"] = df_new["id"].apply(lambda x: clean_id(x, -1))
        df_new["id_руководителя"] = df_new["id_руководителя"].apply(lambda x: clean_id(x, 883))

        # Путь к сохранённому файлу
        output_path = os.path.join(DATA_RAW, "director.csv")
        output_path_history = os.path.join(DATA_RAW, "director_history.csv")

        # Если файл уже существует, читаем старую таблицу
        if os.path.exists(output_path):
            df_old = pd.read_csv(output_path)
        else:
            df_old = pd.DataFrame(columns=columns)

        # Обновляем или добавляем записи
        df_old.set_index("id", inplace=True)
        df_new.set_index("id", inplace=True)

        for idx, row in df_new.iterrows():
            if idx in df_old.index:
                old_dir_id = df_old.at[idx, "id_руководителя"]
                new_dir_id = row["id_руководителя"]
                new_dir_fio = row["фио_руководителя"]

                # Обновляем, если изменился id_руководителя и он не пустой
                if old_dir_id != new_dir_id and new_dir_id != 883 and new_dir_fio != "":
                    df_old.at[idx, "id_руководителя"] = new_dir_id
                    df_old.at[idx, "фио_руководителя"] = new_dir_fio
            else:
                # Новый сотрудник — добавляем полностью
                df_old.loc[idx] = row  # type: ignore

        df_old.reset_index(inplace=True)
        df_old.to_csv(output_path, index=False, sep=",")
        df_old.to_csv(output_path_history, index=False, sep=",")
        logger.info(f"Обновлённые данные сохранены в {output_path}.")

    except Exception as e:
        logger.exception(f"Ошибка при загрузке данных: {e}")
        raise

    finally:
        if driver:
            driver.quit()

    logger.info("Загрузка руководителей пользователей с портала завершена.")


def get_whisper_stat(base_dir, check_list_file, output_file):
    logger.info("Загрузка статистики топов")

    # Загружаем check_list
    check_list = set()
    if os.path.exists(check_list_file):
        with codecs.open(check_list_file, "r", encoding="utf-8-sig") as f:
            check_list = set(os.path.abspath(line.strip()) for line in f if line.strip())

    # Получаем список всех .txt файлов
    all_files = {
        os.path.abspath(os.path.join(root, file))
        for root, _, files in os.walk(base_dir)
        for file in files
        if file.endswith(".txt")
    }

    files_to_process = list(all_files - check_list)

    results = []
    new_check_entries = []

    max_workers = max(1, os.cpu_count() // 2)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in files_to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка файлов"):
            file = futures[future]
            abs_path = os.path.abspath(file)
            try:
                result = future.result()
                if result is not None:
                    row, _ = result
                    results.append(row)
            except Exception:
                pass  # Ошибки молча игнорируем
            finally:
                new_check_entries.append(abs_path)

    # Запись результатов
    if results:
        file_exists = os.path.exists(output_file)
        write_header = not file_exists or os.path.getsize(output_file) == 0

        with open(output_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    # Запись новых путей в check_list
    if new_check_entries:
        with codecs.open(check_list_file, "a", encoding="utf-8-sig") as f:
            for path in new_check_entries:
                f.write(path + "\n")

    logger.info("Загрузка статистики топов завершена")


def process_file(file_path):
    try:
        summ_check = 0
        dialog_analysis = {key: "нет" for key in WHISPER_CATEGORIES}
        dialog_analysis["Тон"] = "дружественный"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.lower().strip() for line in f.readlines()[:8]]

        for line in lines:
            for cat in WHISPER_CATEGORIES:
                tag = f"[{cat.lower()}]"
                if line.startswith(tag):
                    if cat == "Тон":
                        dialog_analysis["Тон"] = (
                            "негативный" if "негативный" in line else "дружественный"
                        )
                    else:
                        dialog_analysis[cat] = "да" if "да" in line else "нет"

        for cat, val in dialog_analysis.items():
            if val == "да":
                summ_check += WHISPER_CATEGORIES[cat]["да"]

        if summ_check == 0:
            return None

        parts = file_path.split("-")
        username = parts[2].split("@")[0]

        row = {
            "логин": username,
            "тон": WHISPER_CATEGORIES["Тон"][dialog_analysis["Тон"]],
            "увольнение": WHISPER_CATEGORIES["Увольнение"][dialog_analysis["Увольнение"]],
            "оффер": WHISPER_CATEGORIES["Оффер"][dialog_analysis["Оффер"]],
            "вредительство": WHISPER_CATEGORIES["Вредительство"][dialog_analysis["Вредительство"]],
            "личная жизнь": WHISPER_CATEGORIES["Личная жизнь"][dialog_analysis["Личная жизнь"]],
            "стресс": WHISPER_CATEGORIES["Стресс"][dialog_analysis["Стресс"]],
            "конфликты": WHISPER_CATEGORIES["Конфликты"][dialog_analysis["Конфликты"]],
        }

        return row, file_path  # Вернём row + info для check_list

    except Exception as e:
        logger.error(f"Ошибка в файле {file_path}: {e}")
        return None


def get_1c_zup(base_name, server_1c, login, password):
    try:
        logger.info("Загрузка ЗУП")
        logger.info("Загрузка ЗУП завершена")
    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def run_all():
    get_latest_file(cadr_users_list_url)
    get_portal_users(portal_users_link)
    get_portal_children(portal_children_link)
    get_portal_director(portal_director_link)
    get_whisper_stat(
        base_dir=whisper_data,
        check_list_file=os.path.join(DATA_RAW, "check_list.txt"),
        output_file=os.path.join(DATA_RAW, "whisper_stat.csv"),
    )
    get_1c_zup(Ref_zup, Srvr_zup, login_1c, password_1c)
    # pass


if __name__ == "__main__":
    run_all()
