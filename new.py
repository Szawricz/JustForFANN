from random import uniform

from numpy import exp


def sig(neuro_sum: float):
    """Return the sigmoid function result.

    Args:
        neuro_sum: a weighted sum of neurons

    Returns:
        The return value in float type

    """
    return 2 / (1 + exp(-neuro_sum)) - 1


class Weight():
    def __init__(self):
        self.value = uniform(-1, 1)

    def __repr__(self):
        return f'< Weight: {self.value} >'


class Neuron:
    def __init__(self, inputs_number: int, transmition_function=sig):
        self.transmition_function = transmition_function
        self.weights = [Weight() for weight in range(inputs_number)]

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
        transmition_function=sig,
    ):
        self.neurons = [BiasNeuron()]
        for neuron_number in range(neurons_number-1):
            neuron = Neuron(
                inputs_number=last_layer_neurons_number,
                transmition_function=transmition_function,
            )
            self.neurons.append(neuron)

    def get_outputs(self, inputs_values: list) -> list:
        return [neuron.get_output(inputs_values) for neuron in self.neurons]


class Perceptron:
    def __init__(self, structure: list, transmition_function=sig):
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
                ),
            )

    def get_outputs(self, inputs_values: list):
        inputs_values.insert(0, 1)
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer.get_outputs(resoults)
        return resoults[1:]


