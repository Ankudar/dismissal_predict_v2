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


def get_portal_credentials():
    config = load_config()
    login = config["portal_root"]["login"]
    password = config["portal_root"]["password"]
    return login, password


def get_cadr_users_list():
    config = load_config()
    url = config["links"]["cadr_users_link"]
    return url


def get_portal_children_link():
    config = load_config()
    url = config["links"]["portal_children_link"]
    return url


def get_portal_users_link():
    config = load_config()
    url = config["links"]["portal_users_link"]
    return url


def get_1c_info():
    config = load_config()
    login_1c = config["1c_info"]["login"]
    password_1c = config["1c_info"]["password"]
    Srvr_zup = config["1c_info"]["Srvr"]
    Ref_zup = config["1c_info"]["Ref"]
    return login_1c, password_1c, Srvr_zup, Ref_zup


ad_login, ad_password = get_ad_credentials()
portal_login, portal_password = get_portal_credentials()
cadr_users_list_url = get_cadr_users_list()
portal_children_link = get_portal_children_link()
portal_users_link = get_portal_users_link()
login_1c, password_1c, Srvr_zup, Ref_zup = get_1c_info()

if __name__ == "__main__":
    pass
