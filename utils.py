from random import uniform, choice

from numpy import exp


def sig(neuro_sum: float):
    """Return the sigmoid function result.

    Args:
        neuro_sum: a weighted sum of neurons

    Returns:
        The return value in float type

    """
    return 2 / (1 + exp(-neuro_sum)) - 1


def generate_uniform() -> float:
    return uniform(-1, 1)


def generate_sign() -> int:
    return choice([-1, 0, 1])


def make_simple_structure(
    inputs_number: int, intermediate_layers_number: int, outputs_number: int,
) -> list:
    interm_layers_neurons_number = max([inputs_number, outputs_number]) + 1
    resoult_structure = [inputs_number]
    for item in range(intermediate_layers_number):
        resoult_structure.append(interm_layers_neurons_number)
    resoult_structure.append(outputs_number)
    return resoult_structure
