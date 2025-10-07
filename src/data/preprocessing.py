import pandas as pd

def prepare_data(df: pd.DataFrame, mark_thr: float = 0.8, filter_items=5, filter_users=5):
    df = df[df["rating"] > mark_thr]

    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        
        # Фильтрация пользователей
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= filter_users].index
        df = df[df["user_id"].isin(valid_users)]
        
        # Фильтрация айтемов
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= filter_items].index
        df = df[df["item_id"].isin(valid_items)]

    # users = df['user_id'].unique()[:5000]
    # df = df[df['user_id'].isin(users)]

    # items = df['item_id'].unique()[:3000]
    # df = df[df['item_id'].isin(items)]

    num_items = df['item_id'].nunique()
    print('N items', num_items )

    user2id = {val:i for i, val in enumerate(df['user_id'].unique())}
    item2id = {val:i+1 for i, val in enumerate(df['item_id'].unique())}

    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)

    return df, num_items

def get_sequences(dataset: pd.DataFrame, split_ratio: float = 0.8):
    sequences = []
    times = []

    for user_id, group in dataset.groupby('user_id'):
        group = group.sort_values('timestamp')
        user_seq = group['item_id'].tolist()
        if len(user_seq) < 5:
            continue 
        user_times = group['timestamp'].astype(int).tolist()
        sequences.append(user_seq)
        times.append(user_times)

    # Разделяем на train/val
    split_idx = int(split_ratio * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]

    train_times = times[:split_idx]
    val_times = times[split_idx:]

    all_users = dataset['user_id'].unique()
    train_users, val_users = all_users[:split_idx], all_users[split_idx:]
    return (train_sequences, val_sequences), (train_times, val_times), (train_users, val_users) 