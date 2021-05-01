from operator import itemgetter
from random import choice, uniform

from numpy import exp


class NotToughtPopulation(BaseException):
    "The population is not thought yet"
    pass


def sig(neuro_sum: float):
    """Return the sigmoid function result.

    Args:
        neuro_sum: a weighted sum of neurons

    Returns:
        The return value in float type

    """
    return 2 / (1 + exp(-neuro_sum)) - 1


def relu(neuro_sum: float):
    if neuro_sum < 0:
        return 0
    return neuro_sum


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
        сalibration_functions=first_neuronet.сalibration_functions,
        transmition_function=first_neuronet.all_neurons[1].transmition_function,
    )


def get_succeses(neuronets, dataset):
    successes = list()
    for neuronet in neuronets:
        dataset_cases_errors = list()
        for dataset_inputs, dataset_outputs in dataset:
            outputs = neuronet.get_outputs(dataset_inputs)
            output_errors = list()
            for number, value in enumerate(dataset_outputs):
                output_errors.append(abs(value - outputs[number]))
            dataset_cases_errors.append(max(output_errors))
        successes.append(1 - max(dataset_cases_errors) / 2)
    return successes


def get_neuronets_sorted_by_succes(successes, neuronets):
    return list(
        list(
            zip(
                *sorted(
                    zip(successes, neuronets),
                    key=itemgetter(0),
                )
            )
        )[1]
    )
