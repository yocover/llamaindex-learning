# 全局变量，用于存储ETCD配置

from src.utils.etcd_util import ConfigManager
from src.constants import ETCD_HOST, ETCD_PORT, ETCD_PREFIX


ETCD_CONFIG = None


def init_config():
    global ETCD_CONFIG
    instance = EtcdConfig(
        host=ETCD_HOST,
        port=ETCD_PORT,
        prefix=ETCD_PREFIX,
    )
    ETCD_CONFIG = instance.config
    return ETCD_CONFIG


class EtcdConfig:

    def __init__(self, host: str, port: int, prefix: str):
        self.host = host
        self.port = port
        self.prefix = prefix

        self.client = ConfigManager(
            etcd_host=self.host, etcd_port=self.port, config_prefix=self.prefix
        )

        self.config = self.client.get_config()

    def get_config(self, key: str) -> str:
        return self.config.get(key)

    def set_config(self, key: str, value: str):
        self.client.put(self.prefix + key, value)
        self.config = self.client.get_config()

    def set_config(self, key: str, value: str):
        ETCD_CONFIG[key] = value
