from random import choice, uniform

from numpy import exp


def sig(neuro_sum: float) -> float:
    """Return the sigmoid function result.
    Args:
        neuro_sum: a weighted sum of neurons
    Returns:
        The return value in float type
    """
    return 2 / (1 + exp(-neuro_sum)) - 1


def softsign(neuro_sum: float) -> float:
    return neuro_sum / (1 + abs(neuro_sum))


def heaviside(neuro_sum: float) -> int:
    if neuro_sum >= 0:
        return 1
    return 0


def string_to_numbers_list(string: str) -> list:
    return [ord(character) for character in string]


def numbers_list_to_string(numbers_list: list) -> str:
    unicode_max = 1114111
    chars_list = list()
    for number in numbers_list:
        chars_list.append(chr(round((number + 1) / 2 * unicode_max)))
    return str().join(chars_list)


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


def cross_over(first_neuronet, second_neuronet, mutability) -> object:
    compliment_weights_couples = list(
        zip(first_neuronet.all_weights, second_neuronet.all_weights)
    )
    child_raw_weights = list()
    for first_weight, second_weight in compliment_weights_couples:
        if choice([True, False]):
            child_raw_weight = first_weight.value
        else:
            child_raw_weight = second_weight.value
        if uniform(0, 1) < mutability:
            child_raw_weight = first_weight.value_generator()
        child_raw_weights.append(child_raw_weight)
    return first_neuronet.__class__.init_from_weights(
        child_raw_weights, *first_neuronet.essential_attrs,
    )
