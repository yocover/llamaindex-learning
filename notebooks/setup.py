"""
用于 Jupyter notebook 的初始化设置
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import sys
from config.etcd_config import init_config as get_etcd_config


@dataclass
class SiliconFlowConfig:
    api_key: str
    api_base: str


@dataclass
class Config:
    siliconflow: SiliconFlowConfig


def get_setup_settings() -> Config:
    """获取 notebook 环境配置"""
    etcd_config = get_etcd_config()
    siliconflow_config = etcd_config.get("siliconflow")

    silicon_config = SiliconFlowConfig(
        api_base=siliconflow_config.get("api_base"),
        api_key=siliconflow_config.get("api_key"),
    )

    return Config(siliconflow=silicon_config)


def init_setup_settings() -> Config:
    """设置 notebook 环境"""
    # 添加项目根目录到 Python 路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
    config = get_setup_settings()
    return config
