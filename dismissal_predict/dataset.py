from datetime import datetime
import os
import shutil

from config import ad_login, ad_password, cadr_users_list_url

DIR = "data/raw"


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

    return latest_file


if __name__ == "__main__":
    latest_file = get_latest_file(cadr_users_list_url)
    if latest_file:
        destination_file = os.path.join(DIR, "last_users_from_cadr.xls")
        shutil.copy(latest_file, destination_file)
    else:
        print("Файлы не найдены.")
