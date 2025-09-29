import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from src.data import get_sequences, load_data, download_dataset, prepare_data, ValSASRecDataset, TrainSASRecDataset
from src.aggregation import get_external_vectors, transform_aggregation
from src.utils import fix_seed, save_config
from src.model import SASRec, train_sasrec, calculate_ndcg_loss

import warnings
warnings.filterwarnings("ignore")

CONFIG_NAME = "main"
CONFIG_PATH = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    print("Aggregation method", cfg['agg_method'])
    if cfg['save_mode'] == 'test':
        print("!!!!!!!!!!!!!!!!!! TEST MODE! NO SAVED RESULTS !!!!!!!!!!!!!!!!!!!!")
    else:
        file_name = f"{cfg['exp_name']}|model={'sasrec'}_seed={cfg['seed']}_final={cfg['final']}|agg={cfg['agg_method']}__head={cfg['head_method']}"
        save_config(cfg["checkpoint_path"], CONFIG_PATH, CONFIG_NAME, file_name)
    
    fix_seed(cfg["seed"])
    
    # Скачиваем и загружаем данные
    download_dataset(cfg['data_path'], cfg['dataset'])
    df = load_data(cfg['data_path'], cfg['dataset'])

    df, num_items = prepare_data(df, cfg['mark_thr'], cfg['filter_items'], cfg['filter_users'])

    (train_sequences, val_sequences), (train_times, val_times), (train_users, val_users) = get_sequences(df)
    train_dataset = TrainSASRecDataset(train_sequences, train_times, cfg["max_len"])
    val_dataset = ValSASRecDataset(val_sequences, val_times, cfg["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["val_batch_size"], shuffle=False)


    # Инициализируем модель
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = SASRec(
        num_items=num_items,
        embedding_size=cfg["embedding_size"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        dropout_rate=cfg["dropout_rate"],
        final=cfg['final'],
        max_len=cfg["max_len"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr_pretrain"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_name = f"{cfg['exp_name']}|model={'sasrec'}_seed={cfg['seed']}_final={cfg['final']}"
    metrics_name = f"{checkpoint_name}.csv"
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
            save_mode=cfg['save_mode'],
            epochs=cfg["n_pretrain_epochs"],
        )

    if cfg['run_aggregation']:
        checkpoint_name = f"{cfg['preload_checkpoint']}.pth" if cfg['preload_checkpoint'] else f"{checkpoint_name}.pth"
        load_path = os.path.join(cfg["checkpoint_path"], checkpoint_name)
        model.load_state_dict(torch.load(load_path))

        user_embeddings, all_timestamps = get_external_vectors(model, device, df, train_users, cfg["n_ext_users"])
        time_list, time_to_embeddings, ext_embeddings = transform_aggregation(all_timestamps, user_embeddings, cfg["n_ext_users"], cfg["embedding_size"])
        try:
            model.add_external_features(
                time_list,
                time_to_embeddings,
                ext_embeddings,
                head_method=cfg["head_method"],
                agg_type=cfg["agg_method"],
                freezing=cfg['freeze'],
                alpha=cfg['alpha'],
                additional_config=cfg["aggregations"][cfg["agg_method"]]["additional_config"]
            )
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr_exttrain"])
            criterion = nn.CrossEntropyLoss()

            checkpoint_name = f"{cfg['exp_name']}|model={'sasrec'}_seed={cfg['seed']}_final={cfg['final']}|agg={cfg['agg_method']}__head={cfg['head_method']}__freeze={cfg['freeze']}"
            metrics_name = f"{checkpoint_name}.csv"
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
                learnable=cfg["aggregations"][cfg["agg_method"]]['learnable'],
                checkpoint_name=checkpoint_name,
                metrics_name=metrics_name,
                save_mode=cfg['save_mode'],
                epochs=cfg["n_exttrain_epochs"]
            )
        except Exception as e: 
            print(e)
            print("Fail")



if __name__ == "__main__":
    main()