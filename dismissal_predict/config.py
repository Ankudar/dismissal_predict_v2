import json
from pathlib import Path

MAIN_CONFIGS = "/home/root6/python/main_config.json"


class Config:
    WHISPER_CATEGORIES = {
        "Увольнение": {"нет": 0, "да": 1},
        "Оффер": {"нет": 0, "да": 1},
        "Вредительство": {"нет": 0, "да": 1},
        "Конфликты": {"нет": 0, "да": 1},
        "Стресс": {"нет": 0, "да": 1},
        "Личная жизнь": {"нет": 0, "да": 1},
        "Тон": {"дружественный": 0, "негативный": 1},
    }

    WHISPER_CATEGORIES_WEIGHT = {
        "увольнение": {0: 0, 1: 70},
        "оффер": {0: 0, 1: 60},
        "вредительство": {0: 0, 1: 50},
        "конфликты": {0: 0, 1: 40},
        "стресс": {0: 0, 1: 30},
        "личная жизнь": {0: 0, 1: 20},
        "тон": {0: 0, 1: 10},
    }

    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

    def get_ad_credentials(self):
        login = self.config["ad_root"]["login"]
        password = self.config["ad_root"]["password"]
        return login, password

    def get_portal_credentials(self):
        login = self.config["portal_root"]["login"]
        password = self.config["portal_root"]["password"]
        return login, password

    def get_cadr_users_list(self):
        url = self.config["links"]["cadr_users_link"]
        return url

    def get_portal_children_link(self):
        url = self.config["links"]["portal_children_link"]
        return url

    def get_portal_users_link(self):
        url = self.config["links"]["portal_users_link"]
        return url

    def get_whisper_url(self):
        url = self.config["links"]["whisper_data"]
        return url

    def get_1c_info(self):
        login = self.config["1c_info"]["login"]
        password = self.config["1c_info"]["password"]
        server = self.config["1c_info"]["Srvr"]
        base = self.config["1c_info"]["Ref"]
        return login, password, server, base


if __name__ == "__main__":
    pass
