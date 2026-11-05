import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BIT_COUNT = 20
SAMPLE_COUNT = 30_000

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def create_dataset():
    # Fiecare identificator reprezinta o combinatie unica de senzori.
    identifiers = rng.choice(
        2 ** BIT_COUNT,
        size=SAMPLE_COUNT,
        replace=False,
    )

    front = identifiers & 31
    left = (identifiers >> 5) & 31
    right = (identifiers >> 10) & 31
    speed = (identifiers >> 15) & 15
    wet = (identifiers >> 19) & 1

    front_threshold = 2 + speed + wet * (speed // 2)

    must_brake = (
        (front <= front_threshold)
        | (
            (speed >= 8)
            & (np.minimum(left, right) <= 2)
        )
    )

    # Bitii sunt ordonati de la cel mai putin semnificativ
    bits = (
        identifiers[:, None] >> np.arange(BIT_COUNT)
    ) & 1

    inputs = torch.tensor(
        2 * bits - 1,
        dtype=torch.float32,
    )

    labels = torch.tensor(
        must_brake.astype(np.float32),
    ).reshape(-1, 1)

    return identifiers, inputs, labels


identifiers, inputs, labels = create_dataset()

train_end = 21_000
validation_end = 25_500

train_dataset = TensorDataset(
    inputs[:train_end],
    labels[:train_end],
)

validation_inputs = inputs[train_end:validation_end]
validation_labels = labels[train_end:validation_end]

test_inputs = inputs[validation_end:]
test_labels = labels[validation_end:]

print("Total examples:", len(inputs))
print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_inputs))
print("Test examples:", len(test_inputs))
print("Brake proportion:", labels.mean().item())