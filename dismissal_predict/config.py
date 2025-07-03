import json
from pathlib import Path

MAIN_CONFIGS = Path("D:/python/VSCode/main_config.json")


def load_config():
    with open(MAIN_CONFIGS, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config


def get_ad_credentials():
    config = load_config()
    login = config["ad_root"]["login"]
    password = config["ad_root"]["password"]
    return login, password


def get_cadr_users_list():
    config = load_config()
    url = config["links"]["cadr_users_link"]
    return url


ad_login, ad_password = get_ad_credentials()
cadr_users_list_url = get_cadr_users_list()

if __name__ == "__main__":
    pass
