import argparse
import json
from pathlib import Path
from time import perf_counter

import z3


def load_network(model_path):
    with model_path.open(encoding="utf-8") as file:
        network = json.load(file)

    # Verificăm structura modelului exportat.
    expected_inputs = 20
    layers = network["layers"]

    if not layers:
        raise ValueError("Modelul nu contine straturi.")

    for index, layer in enumerate(layers):
        weights = layer["weights"]
        biases = layer["biases"]

        if not weights or len(weights) != len(biases):
            raise ValueError(f"Structura invalida in stratul {index}.")

        for row in weights:
            if len(row) != expected_inputs:
                raise ValueError(
                    f"Numar incorect de ponderi in stratul {index}."
                )

            if any(weight not in (-1, 1) for weight in row):
                raise ValueError(
                    f"Ponderi nebinarizate in stratul {index}."
                )

        if any(bias != int(bias) for bias in biases):
            raise ValueError(
                f"Biasurile trebuie sa fie intregi: stratul {index}."
            )

        expected_inputs = len(weights)

    if expected_inputs != 1:
        raise ValueError("Reteaua trebuie sa aiba exact o iesire.")

    return network


def evaluate_python(network, bits):
    """Evaluare directa a retelei exportate, fara Z3."""
    values = [1 if bit else -1 for bit in bits]

    for layer_index, layer in enumerate(network["layers"]):
        scores = [
            sum(
                int(weight) * value
                for weight, value in zip(weights, values)
            ) + int(bias)
            for weights, bias in zip(
                layer["weights"],
                layer["biases"],
            )
        ]

        is_output_layer = (
            layer_index == len(network["layers"]) - 1
        )

        if is_output_layer:
            return scores[0]

        values = [1 if score >= 0 else -1 for score in scores]

    raise ValueError("Reteaua nu contine un strat de iesire.")


def decode_python(bits, start, width):
    return sum(
        bits[start + offset] * (2 ** offset)
        for offset in range(width)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cauta contraexemple pentru o BNN folosind Z3."
    )

    parser.add_argument(
        "architecture",
        choices=["small", "medium", "initial", "large"],
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60_000,
        help="Limita de timp pentru Z3, in milisecunde.",
    )

    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout trebuie sa fie pozitiv.")

    model_path = (
        Path(__file__).resolve().parent
        / "artifacts"
        / args.architecture
        / "bnn.json"
    )

    network = load_network(model_path)

    print("Architecture:", args.architecture)
    print("Model file:", model_path)

    solver = z3.Solver()
    solver.set(timeout=args.timeout)

    # Cei 20 de biti codifica valorile celor cinci senzori.
    input_bits = [
        z3.Bool(f"input_{index}")
        for index in range(20)
    ]

    # Bitul 0 devine -1, iar bitul 1 devine +1.
    values = [
        z3.If(bit, 1, -1)
        for bit in input_bits
    ]

    output_score = None

    # Construim reprezentarea simbolica a retelei.
    # Ponderile si biasurile sunt constantele din JSON.
    for layer_index, layer in enumerate(network["layers"]):
        next_values = []

        is_output_layer = (
            layer_index == len(network["layers"]) - 1
        )

        for neuron_index, (weights, bias) in enumerate(
            zip(layer["weights"], layer["biases"])
        ):
            score = z3.Sum([
                int(weight) * value
                for weight, value in zip(weights, values)
            ]) + int(bias)

            if is_output_layer:
                output_score = score
            else:
                activation = z3.Int(
                    f"layer_{layer_index}_neuron_{neuron_index}"
                )

                solver.add(
                    activation == z3.If(score >= 0, 1, -1)
                )

                next_values.append(activation)

        values = next_values

    def decode_sensor(start, width):
        return z3.Sum([
            z3.If(
                input_bits[start + offset],
                2 ** offset,
                0,
            )
            for offset in range(width)
        ])

    front = decode_sensor(0, 5)
    left = decode_sensor(5, 5)
    right = decode_sensor(10, 5)
    speed = decode_sensor(15, 4)
    wet = decode_sensor(19, 1)

    # Aceeasi regula folosita pentru etichetele de antrenare.
    # speed este intreg nenegativ: impartirea Z3 este intreaga.
    front_threshold = (
        2 + speed + z3.If(wet == 1, speed / 2, 0)
    )

    must_brake = z3.Or(
        front <= front_threshold,
        z3.And(
            speed >= 8,
            z3.Or(left <= 2, right <= 2),
        ),
    )

    # Contraexemplu:
    # regula cere franare, dar reteaua decide sa continue.
    # Scor >= 0 inseamna franare; scor < 0 inseamna continuare.
    solver.add(must_brake)
    solver.add(output_score < 0)

    started = perf_counter()
    result = solver.check()
    elapsed = perf_counter() - started

    print("Z3 result:", result)
    print(f"Solver time: {elapsed:.6f} seconds")

    if result == z3.sat:
        solution = solver.model()

        bits = [
            int(z3.is_true(
                solution.eval(bit, model_completion=True)
            ))
            for bit in input_bits
        ]

        sensors = {
            "front": decode_python(bits, 0, 5),
            "left": decode_python(bits, 5, 5),
            "right": decode_python(bits, 10, 5),
            "speed": decode_python(bits, 15, 4),
            "wet": decode_python(bits, 19, 1),
        }

        # Verificam independent regula, folosind Python.
        python_threshold = (
            2
            + sensors["speed"]
            + sensors["wet"] * (sensors["speed"] // 2)
        )

        python_must_brake = (
            sensors["front"] <= python_threshold
            or (
                sensors["speed"] >= 8
                and min(
                    sensors["left"],
                    sensors["right"],
                ) <= 2
            )
        )

        # Verificam independent si decizia retelei.
        python_score = evaluate_python(network, bits)
        z3_score = solution.eval(output_score).as_long()

        print("Counterexample:", sensors)
        print("Rule requires braking:", python_must_brake)
        print("Python output score:", python_score)
        print("Z3 output score:", z3_score)
        print(
            "Network decision:",
            "BRAKE" if python_score >= 0 else "CONTINUE",
        )

        if python_score != z3_score:
            raise RuntimeError(
                "Scorul Python difera de scorul calculat de Z3."
            )

        if not python_must_brake or python_score >= 0:
            raise RuntimeError(
                "Solutia gasita nu este un contraexemplu valid."
            )

        print("Python verification passed.")

    elif result == z3.unsat:
        print(
            "Nu exista contraexemple pentru proprietatea "
            "de franare in domeniul codificat."
        )

    else:
        print("Z3 nu a putut stabili rezultatul.")
        print("Reason:", solver.reason_unknown())


if __name__ == "__main__":
    main()