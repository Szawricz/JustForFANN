from random import choice, uniform


class NotToughtPopulation(BaseException):
    "The population is not thought yet"
    pass


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
        weights=child_raw_weights,
        structure=first_neuronet.structure,
    )
