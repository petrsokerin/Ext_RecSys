from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch


def get_external_vectors(model, device, dataset, train_users, n_ext_users):
    user_subset = np.random.choice(train_users, n_ext_users, replace=False)
    subset_data = dataset[dataset['user_id'].isin(user_subset)].sort_values(['user_id', 'timestamp'])

    # Создаем словарь для хранения эмбеддингов пользователей по времени
    user_embeddings = defaultdict(list)
    #time_embeddings = defaultdict(list)
    all_timestamps = set()

    # Получаем эмбеддинги для каждого пользователя в каждый момент времени
    model.eval()
    with torch.no_grad():
        for user_id in tqdm(user_subset, desc="Processing users"):
            user_data = subset_data[subset_data['user_id'] == user_id]
            user_items = user_data['item_id'].tolist()
            timestamps = user_data['timestamp'].tolist()
            
            if len(user_items) <= model.max_len:
                padding_mask = torch.ones(model.max_len, dtype=torch.bool)
                n_pads = model.max_len - len(user_items)               
                seq = [0] * (n_pads) + user_items
                padding_mask[:n_pads] = False
                input_seq = torch.tensor([seq], dtype=torch.long).to(device)
                padding_mask = padding_mask.to(device)
                embeddings = model.get_internal_embeddings(input_seq, padding_mask).cpu().numpy()[0, :, :]
            else:
                for i in range(0, len(user_items) + 1, model.max_len):
                    seq = user_items[i:model.max_len + i]
                    padding_mask = torch.ones(len(seq), dtype=torch.bool)
                    input_seq = torch.tensor([seq], dtype=torch.long).to(device)
                    padding_mask = padding_mask.to(device)
                    batch_emb = model.get_internal_embeddings(input_seq, padding_mask).cpu().numpy()[0, :, :]
                    if i > 0:
                        embeddings = np.concatenate([embeddings, batch_emb], axis=0)
                        all_pad_mask = np.concatenate([embeddings, batch_emb], axis=0)
                    else:
                        embeddings = batch_emb

            for time, embed in zip(timestamps, embeddings):
                all_timestamps.add(time)
                user_embeddings[user_id].append((time, embed))
                #time_embeddings[time].append((user_id, embed))

    all_timestamps = sorted(list(all_timestamps))

    return user_embeddings, all_timestamps

def transform_aggregation(all_timestamps, user_embeddings, n_ext_users, embedding_size):
    agg_raw_data_embeddings = []

    min_timestamp = np.min(all_timestamps)

    for timestamp in tqdm(all_timestamps, desc="Aggregating embeddings"):
        # Собираем последние эмбеддинги для каждого пользователя на данный момент времени
        current_embeddings = []
        current_times = []
        for user_id, embeddings_list in user_embeddings.items():
            # Находим последний эмбеддинг пользователя до данного времени
            user_embs_before = [(ts, emb) for ts, emb in embeddings_list if ts <= timestamp]
            if user_embs_before:
                # Берем последний эмбеддинг
                time, last_emb = user_embs_before[-1]
                current_embeddings.append(np.array(last_emb))
                current_times.append(time)

        # Агрегируем эмбеддинги всех пользователей
        if len(current_embeddings) < n_ext_users:
            n_embeddings = len(current_embeddings)
            for i in range(n_ext_users - n_embeddings):
                current_embeddings.append(np.zeros(embedding_size))
                current_times.append(min_timestamp)
        agg_raw_data_embeddings.append([timestamp, current_times, current_embeddings])


    agg_raw_data_embeddings = sorted(agg_raw_data_embeddings, key = lambda d: d[0])

    time_list = np.array([data[0] for data in agg_raw_data_embeddings])
    time_to_embeddings = np.array([data[1] for data in agg_raw_data_embeddings])
    ext_embeddings = np.array([data[2]  for data in agg_raw_data_embeddings])

    return time_list, time_to_embeddings, ext_embeddings