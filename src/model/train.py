import os

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from tqdm import tqdm


def calculate_ndcg_loss(model, dataloader, criterion, device, k=10):
    model.eval()

    # Вместо накопления всех предсказаний, будем вычислять NDCG по батчам
    ndcg_scores = []

    total_loss = 0

    with torch.no_grad():
        for (input_seqs, input_times), targets in dataloader:
            input_seqs, input_times, targets = input_seqs.to(device), input_times.to(device), targets.to(device)
            outputs = model(input_seqs, input_times)

            last_outputs = outputs[:, -1, :]

            loss = criterion(last_outputs, targets)
            total_loss += loss.item()

            # Преобразуем в numpy
            predictions = last_outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Создаем матрицу релевантности только для текущего батча
            n_items = predictions.shape[1]
            relevance = np.zeros((len(targets_np), n_items))
            relevance[np.arange(len(targets_np)), targets_np] = 1

            # Вычисляем NDCG для текущего батча
            try:
                batch_ndcg = ndcg_score(relevance, predictions, k=k)
                ndcg_scores.append(batch_ndcg)
            except MemoryError:
                # Если батч слишком большой, разбиваем его на подбатчи
                sub_batch_size = 100
                for i in range(0, len(targets_np), sub_batch_size):
                    end_idx = min(i + sub_batch_size, len(targets_np))
                    sub_relevance = relevance[i:end_idx]
                    sub_predictions = predictions[i:end_idx]
                    sub_ndcg = ndcg_score(sub_relevance, sub_predictions, k=k)
                    ndcg_scores.append(sub_ndcg)

    # Возвращаем среднее значение NDCG по всем батчам
    return np.mean(ndcg_scores), round(total_loss/len(dataloader), 4)

# Обучение модели SASRec
def train_sasrec(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_items,
    checkpoint_path,
    mode="pretrain",
    checkpoint_name="",
    metrics_name="",
    epochs=10
):
    model.to(device)
    best_ndcg = 0
    metrics = pd.DataFrame([])
    metric_names = ["Mode", "Epoch", "Train_Loss", "Val_Loss", "Val_NDCG"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, ((input_seqs, input_times), targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_seqs, input_times, targets = input_seqs.to(device), input_times.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(input_seqs, input_times)

            # Получаем предсказания для последнего элемента в последовательности
            # last_outputs = outputs[:, -1, :]
            #loss = criterion(last_outputs, targets)
            outputs, targets = outputs.reshape(-1, num_items+1), targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        train_loss = round(total_loss/len(train_loader), 4)

        # Валидация
        val_ndcg, val_loss = calculate_ndcg_loss(model, val_loader, criterion, device)

        metrics = [mode, epoch, train_loss, val_loss, val_ndcg]
        print(f"Epoch {epoch+1}, Loss: {train_loss}, Val NDCG@{10}: {val_ndcg:.4f}")

        metrics_df = pd.DataFrame.from_dict({metric_name: metric_val for metric_name, metric_val in zip(metric_names, metrics)})
        metrics = pd.concat([metrics, metrics_df])

        # Сохраняем лучшую модель
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{checkpoint_name}.pth"))

    metrics_full_path = os.path.join(checkpoint_path, metrics_name)
    metrics_prev_path = os.path.join(checkpoint_path, metrics_name.split("|")[0])
    if os.path.exists(metrics_prev_path):
        prev_metric = pd.read_csv(metrics_prev_path)
        metrics = pd.concat([prev_metric, metrics])
    metrics.to_csv(metrics_full_path, index=False)

    return model, best_ndcg