from functools import lru_cache, wraps
from pickle import dump, load
from random import choice, uniform
from time import gmtime, strftime, time

from numpy import exp


class PickleMixin:
    def save_to_file(self, file_path: str):
        with open(file_path, 'wb') as fle:
            dump(self, fle)

    @classmethod
    def load_from_file(cls, file_path: str) -> object:
        with open(file_path, 'rb') as fle:
            return load(fle)


@lru_cache()
def sig(neuro_sum: float) -> float:
    """Return the sigmoid function result.
    Args:
        neuro_sum: a weighted sum of neurons
    Returns:
        The return value in float type
    """
    return 2 / (1 + exp(-neuro_sum)) - 1


@lru_cache()
def softsign(neuro_sum: float) -> float:
    return neuro_sum / (1 + abs(neuro_sum))


def heaviside(neuro_sum: float) -> int:
    if neuro_sum >= 0:
        return 1
    return 0


def linear(neuro_sum: float) -> float:
    return neuro_sum


def generate_uniform() -> float:
    return uniform(-1, 1)


def generate_reversed_uniform() -> float:
    return 1 / uniform(-1, 1)


def generate_sign() -> int:
    return choice([-1, 0, 1])


def generate_0_or_1() -> int:
    return choice([0, 1])


def make_simple_structure(
    inputs_number: int, intermediate_layers_number: int,
    intermediate_layers_neurons_number: int, outputs_number: int,
) -> list:
    interm_layers_neurons_number = intermediate_layers_neurons_number
    resoult_structure = [inputs_number]
    for _item in range(intermediate_layers_number):
        resoult_structure.append(interm_layers_neurons_number)
    resoult_structure.append(outputs_number)
    return resoult_structure


@lru_cache()
def time_lenght_str(time: float) -> str:
    return strftime('%X', gmtime(time))


def measure_execution_time(procedure):
    @wraps(procedure)
    def _wrapper(*args, **kwargs) -> float:
        start = time()
        procedure(*args, **kwargs)
        finish = time()
        return finish - start
    return _wrapper
