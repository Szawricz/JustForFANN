from functools import lru_cache, wraps
from pickle import dump, load
from random import choice, uniform
from time import ctime, gmtime, strftime, time

from numpy import exp


class PickleMixin:
    def save_to_pickle(self, file_path: str):
        with open(file_path, 'wb') as fle:
            dump(self, fle)

    @classmethod
    def load_from_pickle(cls, file_path: str) -> object:
        with open(file_path, 'rb') as fle:
            return load(fle)


class JsonMixin:
    def save_to_json(self, file_path: str):
        with open(file_path, 'wb') as fle:
            dump(self, fle)

    @classmethod
    def load_from_json(cls, file_path: str) -> object:
        with open(file_path, 'rb') as fle:
            return load(fle)


@lru_cache()
def sig(neuro_sum: float) -> float:
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


def with_start_and_finish_time_print(function):
    @wraps(function)
    def _wrapper(*args, **kwargs) -> float:
        start_time = time()
        print(f'Started at: {ctime(start_time)}')
        print(79 * '=')
        resoult = function(*args, **kwargs)
        finish_time = time()
        print(79 * '=')
        print(f'Finished at: {ctime(finish_time)}')
        print(f'TOTAL TIME: {ctime(finish_time - start_time)}')
        return resoult
    return _wrapper


def print_spases_line():
    print('\r', 78 * ' ', end='')


def print_percent(name, number, sequence):
    percent = round(number * 100 / len(sequence))
    print(f'\r{name} {percent}%', end=' | ')


def with_current_process_print(string_to_print: str):
    def decorator(function):
        @wraps(function)
        def _wrapper(*args, **kwargs):
            print(f'\r{string_to_print} ', end=' | ')
            resoult = function(*args, **kwargs)
            print_spases_line()
            return resoult
        return _wrapper
    return decorator


def mix_in(*mixins):
    def decorator(class_be_decorated):
        for mixin in mixins:
            callable_attributes = dict()
            for key, value in mixin.__dict__.items():
                if callable(value):
                    callable_attributes[key] = value
                elif isinstance(value, classmethod):
                    callable_attributes[key] = value
            for key, value in callable_attributes.items():
                setattr(class_be_decorated, key, value)
        return class_be_decorated
    return decorator


def pickling(class_be_decorated):
    return mix_in(PickleMixin)(class_be_decorated)


def jsoning(class_be_decorated):
    return mix_in(JsonMixin)(class_be_decorated)
