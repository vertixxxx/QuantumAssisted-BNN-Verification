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

def binary_value(values):
    return torch.where(
        values >= 0,
        torch.ones_like(values),
        -torch.ones_like(values),
    )


def binary_with_gradient(values):
    clipped = values.clamp(-1, 1)
    binary = binary_value(values)

    # Forward: valori -1/+1.
    # Backward: gradientul aproximativ al funcției clamp.
    return clipped + (binary - clipped).detach()


def integer_with_gradient(values):
    # Forward: bias întreg.
    # Backward: gradientul trece prin parametrul real.
    return values + (values.round() - values).detach()


class BinaryLinear(nn.Module):
    def __init__(self, input_count, output_count):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(output_count, input_count)
        )
        self.bias = nn.Parameter(torch.zeros(output_count))

        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, inputs):
        weights = binary_with_gradient(self.weight)
        biases = integer_with_gradient(self.bias)

        return nn.functional.linear(inputs, weights, biases)


class BNN(nn.Module):
    def __init__(self, hidden_sizes=(32, 16)):
        super().__init__()

        dimensions = [BIT_COUNT, *hidden_sizes, 1]

        self.layers = nn.ModuleList(
            BinaryLinear(input_count, output_count)
            for input_count, output_count
            in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, inputs):
        values = inputs

        for layer in self.layers[:-1]:
            values = binary_with_gradient(layer(values))

        # Scorul final este folosit la calcularea pierderii.
        return self.layers[-1](values)


#print("Output shape:", model(inputs[:8]).shape)

@torch.no_grad()
def evaluate(model, evaluation_inputs, evaluation_labels):
    model.eval()

    predictions = model(evaluation_inputs) >= 0
    expected = evaluation_labels.bool()

    accuracy = (
        predictions == expected
    ).float().mean().item()

    dangerous_errors = (
        expected & ~predictions
    ).sum().item()

    required_braking = expected.sum().item()

    missed_braking_rate = (
        dangerous_errors / max(required_braking, 1)
    )

    return {
        "accuracy": accuracy,
        "dangerous_errors": dangerous_errors,
        "missed_braking_rate": missed_braking_rate,
    }



ARCHITECTURES = {
    "small": (8,),
    "medium": (16, 8),
    "initial": (32, 16),
    "large": (64, 32),
}

EPOCHS = 100

# Folderul artifacts va fi creat lângă experiment.py.
output_root = Path(__file__).resolve().parent / "artifacts"
output_root.mkdir(exist_ok=True)

# Toate arhitecturile folosesc exact aceleași date.
np.savez(
    output_root / "dataset_split.npz",
    train_ids=identifiers[:train_end],
    validation_ids=identifiers[train_end:validation_end],
    test_ids=identifiers[validation_end:],
)

all_results = []

for name, hidden_sizes in ARCHITECTURES.items():
    print(f"\nTraining architecture: {name}")
    print(f"Layers: {[BIT_COUNT, *hidden_sizes, 1]}")

    # Resetăm seed-ul înainte de crearea fiecărui model.
    torch.manual_seed(SEED)

    model = BNN(hidden_sizes=hidden_sizes)

    # Resetăm și generatorul pentru ordinea exemplelor.
    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Fiecare model primește un optimizator nou.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    loss_function = nn.BCEWithLogitsLoss()

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()

            scores = model(batch_inputs)
            loss = loss_function(scores, batch_labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.clamp_(-1, 1)

            total_loss += loss.item() * len(batch_inputs)

        validation_metrics = evaluate(
            model,
            validation_inputs,
            validation_labels,
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0:
            average_loss = total_loss / len(train_dataset)

            print(
                f"[{name}] Epoch {epoch:3d} | "
                f"loss={average_loss:.4f} | "
                f"validation={validation_metrics}"
            )

    # Recuperăm versiunea aleasă pe baza validării.
    model.load_state_dict(best_state)

    # Evaluăm această versiune pe setul de test.
    test_metrics = evaluate(
        model,
        test_inputs,
        test_labels,
    )

    print(f"[{name}] Best epoch: {best_epoch}")
    print(f"[{name}] Test results: {test_metrics}")

    # Fiecare arhitectură are propriul folder.
    model_directory = output_root / name
    model_directory.mkdir(exist_ok=True)

    exported_layers = []

    for layer in model.layers:
        exported_layers.append({
            "weights": binary_value(
                layer.weight.detach()
            ).to(torch.int64).tolist(),

            "biases": layer.bias.detach().round().to(
                torch.int64
            ).tolist(),
        })

    exported_model = {
        "name": name,
        "seed": SEED,
        "architecture": [BIT_COUNT, *hidden_sizes, 1],
        "input_encoding": (
            "least significant bit first; 0 -> -1, 1 -> +1"
        ),
        "sensor_order": [
            "front", "left", "right", "speed", "wet"
        ],
        "sensor_bit_widths": [5, 5, 5, 4, 1],
        "hidden_activation": "sum >= 0 -> +1; otherwise -1",
        "output_decision": "sum >= 0 -> brake; otherwise continue",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "layers": exported_layers,
        "test_metrics": test_metrics,
    }

    with (model_directory / "bnn.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(exported_model, file, indent=2)

    torch.save(
        model.state_dict(),
        model_directory / "training_weights.pt",
    )

    all_results.append({
        "name": name,
        "architecture": [BIT_COUNT, *hidden_sizes, 1],
        "best_epoch": best_epoch,
        "validation_accuracy": best_validation_accuracy,
        **test_metrics,
    })

    # Salvăm progresul după fiecare model terminat.
    with (output_root / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(all_results, file, indent=2)

    print(f"Saved to: {model_directory}")


print("\nFinal comparison")

for result in all_results:
    print(
        f"{result['name']:8s} | "
        f"test accuracy={result['accuracy']:.2%} | "
        f"dangerous errors={result['dangerous_errors']} | "
        f"missed braking={result['missed_braking_rate']:.2%}"
    )