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


def generate_integer() -> int:
    return choice([-1, 0, 1])


