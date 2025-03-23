# run_test.py (在项目根目录)
from src.utils.etcd_util_test import get_config_from_etcd, set_config_to_etcd
import json

if __name__ == "__main__":
    # main()
    config = get_config_from_etcd()
    # config = set_config_to_etcd()

    # 格式化打印
    print(json.dumps(config, indent=2))
