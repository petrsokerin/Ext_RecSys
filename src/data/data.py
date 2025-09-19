import requests
import os
import zipfile

import pandas as pd


def download_movielens1m(data_path):
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("Downloading MovieLens 1M dataset...")
        response = requests.get(url, stream=True)
        with open(os.path.join(data_path, "ml-1m.zip"), "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(os.path.join(data_path, "ml-1m.zip"), 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")

# Загрузка данных
def load_data(data_path):
    ratings = pd.read_csv(os.path.join(data_path, "ml-1m", 'ratings.dat'), sep='::', engine='python',
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
    ratings['timestamp'] = ratings['timestamp'] / 1e6
    # ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    return ratings