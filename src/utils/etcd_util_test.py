# app/utils/etcd_util_test.py
import os
import sys
import json

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.utils.etcd_util import ConfigManager
from src.constants import ETCD_HOST, ETCD_PORT, ETCD_PREFIX


def test_etcd_util():
    # 初始化配置管理器
    config_manager = ConfigManager(
        etcd_host=ETCD_HOST, etcd_port=ETCD_PORT, config_prefix=ETCD_PREFIX
    )
    # 首先清理已有配置
    print("Clearing existing configs...")
    if config_manager.clear_config():
        print("Successfully cleared all existing configs")
    else:
        print("Failed to clear configs")

    # 构建配置文件的完整路径
    config_file = os.path.join(project_root, "config", "project-dev.yml")
    print(f"Loading config from: {config_file}")

    # 上传配置文件
    success = config_manager.upload_yaml_config(config_file)
    if success:
        print("Config uploaded successfully")
    else:
        print("Failed to upload config")

    # 获取完整配置
    all_config = config_manager.get_config()
    print("\nAll Config:")
    print(json.dumps(all_config, indent=2))

    # 获取特定配置值
    milvus_host = config_manager.get_value("milvus.milvusHost")
    print(f"\nMilvus Host: {milvus_host}")

    pg_config = config_manager.get_value("postgresql")
    print("\nPostgreSQL Config:")
    print(json.dumps(pg_config, indent=2))


def get_config_from_etcd(host=ETCD_HOST, port=ETCD_PORT, prefix=ETCD_PREFIX):
    config_manager = ConfigManager(etcd_host=host, etcd_port=port, config_prefix=prefix)
    return config_manager.get_config()


def set_config_to_etcd(host=ETCD_HOST, port=ETCD_PORT, prefix=ETCD_PREFIX):
    config_manager = ConfigManager(etcd_host=host, etcd_port=port, config_prefix=prefix)

    config_file = os.path.join(project_root, "src/util", "project-dev.yml")
    success = config_manager.upload_yaml_config(config_file)

    if success:
        print("Config uploaded successfully")
        return "success"
    else:
        print("Failed to upload config")
        return "failed"


# if __name__ == "__main__":
#     set_config_to_etcd()
