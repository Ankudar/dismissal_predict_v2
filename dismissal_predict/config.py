import json
from pathlib import Path

MAIN_CONFIGS = Path("D:/python/VSCode/main_config.json")


class Config:
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

    def get_1c_info(self):
        login_1c = self.config["1c_info"]["login"]
        password_1c = self.config["1c_info"]["password"]
        Srvr_zup = self.config["1c_info"]["Srvr"]
        Ref_zup = self.config["1c_info"]["Ref"]
        return login_1c, password_1c, Srvr_zup, Ref_zup


if __name__ == "__main__":
    config = Config(MAIN_CONFIGS)
    ad_login, ad_password = config.get_ad_credentials()
    portal_login, portal_password = config.get_portal_credentials()
    cadr_users_list_url = config.get_cadr_users_list()
    portal_children_link = config.get_portal_children_link()
    portal_users_link = config.get_portal_users_link()
    login_1c, password_1c, Srvr_zup, Ref_zup = config.get_1c_info()
