def move_batch_to_device(batch, device):
    for key, value in batch.items():
        batch[key] = value.to(device)

    return batch
