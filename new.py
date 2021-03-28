from numpy import mean

from .utils import generate_uniform, sig


class Weight():
    def __init__(self, value_generator=generate_uniform):
        self.value = value_generator()

    def __repr__(self):
        return f'< Weight: {self.value} >'


class Neuron:
    def __init__(
        self, inputs_number: int, transmition_function=sig,
            value_generator=generate_uniform,
    ):
        self.transmition_function = transmition_function
        self.weights = [
            Weight(value_generator=value_generator) for weight in range(inputs_number)
            ]

    def get_output(self, inputs_values: list) -> float:
        weighted_values = list()
        for weight_number, weight in enumerate(self.weights):
            weighted_values.append(inputs_values[weight_number]*weight.value)
        return self.transmition_function(sum(weighted_values))


class BiasNeuron(Neuron):
    def __init__(self):
        super().__init__(inputs_number=0, transmition_function=None)

    def get_output(self, *args) -> int:
        return 1


class Layer:
    def __init__(
        self, last_layer_neurons_number: int, neurons_number: int,
            transmition_function=sig, value_generator=generate_uniform,
    ):
        self.neurons = [BiasNeuron(value_generator=value_generator)]
        for neuron_number in range(neurons_number-1):
            neuron = Neuron(
                inputs_number=last_layer_neurons_number,
                transmition_function=transmition_function,
                value_generator=value_generator,
            )
            self.neurons.append(neuron)

    def get_outputs(self, inputs_values: list) -> list:
        return [neuron.get_output(inputs_values) for neuron in self.neurons]


class Perceptron:
    def __init__(
        self, structure: list, transmition_function=sig,
            value_generator=generate_uniform,
    ):
        structure = [neurons_number + 1 for neurons_number in structure]
        self.layers = list()
        for layer_number, neurons_number in enumerate(structure):
            if layer_number == 0:
                continue
            self.layers.append(
                Layer(
                    neurons_number=neurons_number,
                    last_layer_neurons_number=structure[layer_number - 1],
                    transmition_function=transmition_function,
                    value_generator=value_generator,
                ),
            )

    def get_outputs(self, inputs_values: list):
        inputs_values.insert(0, 1)
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer.get_outputs(resoults)
        return resoults[1:]


class Population:
    def __init__(self, size: int, neuronet_type: type, arguments: list):
        self.neuronets = [neuronet_type(*arguments) for item in range(size)]

    def get_best_neuronet(
        self, dataset: dict, mortality: float,
            success: float, mutability: float,
    ):
        successes = list()
        for neuronet, suc in self.neuronets:
            dataset_parts_errors = list()
            for dict_key, dict_value in dataset.items():
                outputs = neuronet.get_outputs(dict_key)
                atomic_errors = list()
                for number, value in enumerate(dict_value):
                    atomic_errors.append(abs((value - outputs[number]) / value))
                dataset_parts_errors.append(mean(atomic_errors))
            successes.append(1 - mean(dataset_parts_errors))
