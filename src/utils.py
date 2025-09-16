import shutil
import os
from datetime import datetime
import yaml

import random
import numpy as np
import torch 

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_config(path: str, config_path: str, config_name: str, config_save_name: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)

    shutil.copytree(config_path, path + "/config_folder", dirs_exist_ok=True)
    shutil.copyfile(
        f"{config_path}/{config_name}.yaml", path + "/" + config_save_name + '.yaml'
    )

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Создаем словарь с метаданными
    metadata = {"date": date, "time": time}

    # Создаем файл metadata.yaml в указанной директории
    metadata_path = os.path.join(path, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)