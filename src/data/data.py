import requests
import os
import zipfile

import pandas as pd


def download_dataset(data_path, dataset_name):
    if dataset_name == 'movielens_1m':
        download_movielens1m(data_path)
    elif dataset_name == 'beauty':
        donwload_beauty(data_path)
    else:
        raise ValueError("Dataset name not exist")


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

def donwload_beauty(data_path):
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz"
    dataset_name = "beauty"
    raw_file = os.path.join(data_path, f"{dataset_name}.csv.gz")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Downloading dataset {dataset_name}")
        response = requests.get(url, stream=True, verify=False)
        with open(raw_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")


# Загрузка данных
def load_data(data_path, dataset_name='movielens_1m'):
    if dataset_name == 'movielens_1m':
        ratings = pd.read_csv(os.path.join(data_path, "ml-1m", 'ratings.dat'), sep='::', engine='python',
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
        ratings['timestamp'] = ratings['timestamp'] / 1e6
        # ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    elif dataset_name == 'beauty':

        ratings = pd.read_csv(os.path.join(data_path, 'beauty.csv.gz'), compression="gzip")
        ratings = ratings.rename(columns={"parent_asin": "item_id"})
    else:
        raise ValueError("Dataset name not exist")

    return ratings