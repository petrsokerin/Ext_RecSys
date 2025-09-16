import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from src.data import get_sequences, load_data, download_movielens1m, ValSASRecDataset, TrainSASRecDataset
from src.aggregation import get_external_vectors, transform_aggregation
from src.utils import fix_seed, save_config
from src.model import SASRec, train_sasrec

CONFIG_NAME = "main"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    print("Aggregation method", cfg['agg_method'])
    file_name = 'model_{}_{}_agg_{}'.format(
            "sasrec",
            cfg["seed"],
            cfg["agg_method"],
    )

    save_config(cfg["checkpoint_path"], CONFIG_PATH, CONFIG_NAME, file_name)
    fix_seed(cfg["seed"])
    
    # Скачиваем и загружаем данные
    download_movielens1m(cfg['data_path'])
    df = load_data(cfg['data_path'])

    df = df[df["rating"] > 3.5]

    # Подготовка данных
    num_items = df['item_id'].nunique()

    user2id = {val:i for i, val in enumerate(df['user_id'].unique())}
    item2id = {val:i+1 for i, val in enumerate(df['item_id'].unique())}

    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)

    (train_sequences, val_sequences), (train_times, val_times), (train_users, val_users) = get_sequences(df)
    train_dataset = TrainSASRecDataset(train_sequences, train_times, cfg["max_len"])
    val_dataset = ValSASRecDataset(val_sequences, val_times, cfg["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)


    # Инициализируем модель
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = SASRec(
        num_items=num_items,
        embedding_size=cfg["embedding_size"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout_rate=cfg["dropout_rate"],
        max_len=cfg["max_len"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr_pretrain"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_name = f"model_{'sasrec'}_{cfg['seed']}"
    metrics_name = f"model_{'sasrec'}_{cfg['seed']}.csv"
    if cfg['run_pretrain']:
    # Обучаем модель
        model, pretrain_ndcg = train_sasrec(
            model, 
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            num_items,
            checkpoint_path=cfg["checkpoint_path"],
            mode="pretrain",
            checkpoint_name=checkpoint_name,
            metrics_name=metrics_name,
            epochs=cfg["n_pretrain_epochs"],
        )

    if cfg['run_aggregation']:

        model.load_state_dict(torch.load(os.path.join(cfg["checkpoint_path"], f"{checkpoint_name}.pth")))

        user_embeddings, all_timestamps = get_external_vectors(model, device, df, train_users, cfg["n_ext_users"])
        time_list, time_to_embeddings, ext_embeddings = transform_aggregation(all_timestamps, user_embeddings, cfg["n_ext_users"], cfg["embedding_size"])

        model.add_external_features(
            time_list,
            time_to_embeddings,
            ext_embeddings,
            head_method=cfg["head_method"],
            agg_type=cfg["agg_method"],
            additional_config=cfg["aggregations"][cfg["agg_method"]]
        )
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr_exttrain"])
        criterion = nn.CrossEntropyLoss()

        checkpoint_name = f"model_{'sasrec'}_{cfg['seed']}|agg={cfg['agg_method']}"
        metrics_name = f"model_{'sasrec'}_{cfg['seed']}|agg={cfg['agg_method']}.csv"
        model, ext_ndcg = train_sasrec(
            model, 
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            num_items,
            checkpoint_path=cfg["checkpoint_path"],
            mode="external",
            checkpoint_name=checkpoint_name,
            metrics_name=metrics_name,
            epochs=cfg["n_exttrain_epochs"]
        )



if __name__ == "__main__":
    main()