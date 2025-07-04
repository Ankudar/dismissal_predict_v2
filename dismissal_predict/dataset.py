from datetime import datetime
import os
import shutil
import time

from config import (
    Ref_zup,
    Srvr_zup,
    cadr_users_list_url,
    login_1c,
    password_1c,
    portal_children_link,
    portal_login,
    portal_password,
    portal_users_link,
)
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

DATA_RAW = "data/raw"


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
    driver_service = Service(executable_path="D:\\python\\CHROMEDRIVER\\chromedriver.exe")
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


def get_1c_zup(base_name, server_1c, login, password):
    print("Типа выгрузил")


if __name__ == "__main__":
    get_latest_file(cadr_users_list_url)
    get_portal_users(portal_users_link)
    get_portal_children(portal_children_link)
    get_1c_zup(Ref_zup, Srvr_zup, login_1c, password_1c)
