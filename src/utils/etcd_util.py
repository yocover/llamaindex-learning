# app/utils/config_manager.py
import yaml
import etcd3
import json
import logging
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        config_prefix: str = "/config", 
    ):
        """
        初始化配置管理器

        Args:
            etcd_host: ETCD服务器地址
            etcd_port: ETCD服务器端口
            config_prefix: 配置键前缀
        """
        self.etcd_client = etcd3.client(host=etcd_host, port=etcd_port)
        self.config_prefix = config_prefix.rstrip("/")

    def upload_yaml_config(self, yaml_file_path: str) -> bool:
        """
        将整个YAML配置文件作为一个JSON存储到ETCD
        """
        try:
            # 读取YAML文件
            with open(yaml_file_path, "r") as file:
                config = yaml.safe_load(file)

            # 将整个配置转换为JSON字符串
            config_json = json.dumps(config)

            # 存储到ETCD
            self.etcd_client.put(self.config_prefix, config_json)
            logger.info(f"Successfully uploaded config from {yaml_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload config: {e}")
            return False

    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        获取完整的配置
        """
        try:
            value, _ = self.etcd_client.get(self.config_prefix)
            if value is None:
                return None

            return json.loads(value.decode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to get config: {e}")
            return None

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        通过点号路径获取配置值
        例如: 'milvus.milvusHost' 或 'postgresql.username'
        """
        try:
            config = self.get_config()
            if config is None:
                return default

            # 处理嵌套键
            keys = key_path.split(".")
            value = config
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    return default
                value = value[key]

            return value

        except Exception as e:
            logger.error(f"Failed to get value for {key_path}: {e}")
            return default

    def get_all_config(self) -> Dict[str, Any]:
        """
        获取所有配置

        Returns:
            Dict: 所有配置的字典
        """
        try:
            result = {}
            for item in self.etcd_client.get_prefix(self.config_prefix):
                key = (
                    item[1].key.decode("utf-8").replace(f"{self.config_prefix}/", "", 1)
                )
                value = json.loads(item[0].decode("utf-8"))

                # 重建嵌套结构
                current = result
                parts = key.split("/")
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value

            return result

        except Exception as e:
            logger.error(f"Failed to get all config: {e}")
            return {}

    # 删除指定前缀下的所有配置
    def clear_config(self) -> bool:
        """
        删除指定前缀下的所有配置

        Returns:
            bool: 是否成功删除
        """
        try:
            # 删除前缀下的所有键
            self.etcd_client.delete_prefix(self.config_prefix)
            logger.info(f"Successfully cleared all configs under {self.config_prefix}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear configs: {e}")
            return False
