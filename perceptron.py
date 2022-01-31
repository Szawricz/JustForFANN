from sys import float_info

from iteration_utilities import deepflatten

from layer import InputLayer, InternalLayer, OutputLayer
from population import Population


class Perceptron:
    def __init__(self, structure: list):
        self.error = None
        self.structure = structure
        interstructure = [neurons_number + 1 for neurons_number in structure]
        interstructure[-1] -= 1
        self.layers = list()
        for layer_number, neurons_number in enumerate(interstructure):
            if layer_number == 0:
                continue
            if layer_number == 1:
                layer = InputLayer
            elif layer_number == len(interstructure) - 1:
                layer = OutputLayer
            else:
                layer = InternalLayer
            self.layers.append(
                layer(
                    neurons_number=neurons_number,
                    last_layer_neurons_number=interstructure[layer_number - 1],
                ),
            )

    def __repr__(self):
        return f'< Perceptron: {self.structure}> '

    def get_outputs(self, inputs_values):
        inputs_values = list(inputs_values)
        inputs_values.insert(0, 1)
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer.get_outputs(resoults)
        return resoults

    def count_error(self, dataset):
        resoults = list()
        for dataset_inputs, dataset_outputs in dataset:
            resoult_outputs = self.get_outputs(dataset_inputs)
            output_errors = list()
            for waited, real in list(zip(dataset_outputs, resoult_outputs)):
                output_errors.append(abs((waited - real) / 2))
            resoults.append(max(output_errors))
        self.error = max(resoults)

    def tich_by_genetic(
        self, dataset: list, size=100, fertility=2, error=0.25,
        mutability=0.25,
    ) -> object:
        population = Population(size=size-1, neuronet=self)
        population.neuronets.append(self)
        return population.tich(
            dataset=dataset, fertility=fertility,
            error=error, mutability=mutability,
        )

    @classmethod
    def init_from_weights(cls, weights: list, structure: list):
        new_perceptron = cls(structure)
        for position, weight in enumerate(new_perceptron.all_weights):
            weight.value = weights[position]
        return new_perceptron

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def all_weights(self) -> list:
        neurons = self.all_neurons
        weights = list(deepflatten([neuron.weights for neuron in neurons]))
        return weights

    @property
    def all_neurons(self) -> list:
        layers = self.layers
        neurons = list(deepflatten([layer.neurons for layer in layers]))
        return neurons
