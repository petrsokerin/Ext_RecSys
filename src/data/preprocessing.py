import pandas as pd

def get_sequences(dataset: pd.DataFrame, split_ratio: float = 0.8):
    sequences = []
    times = []

    for user_id, group in dataset.groupby('user_id'):
        group = group.sort_values('timestamp')
        user_seq = group['item_id'].tolist()
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