from unpacking_flatten_lists.funcs import niccolum_flatten

from population import Population

from .layer import InputLayer, InternalLayer, OutputLayer


class Perceptron:
    def __init__(self, structure: list):
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

    def tich_by_genetic(
        self, dataset: list, size=100, fertility=2, success=0.75,
        mutability=0.1,
    ) -> object:
        population = Population(
            size=size - 1,
            neuronet_type=self.__class__,
            arguments=dict(
                structure=self.structure,
                transmition_function=self.all_neurons[1].transmition_function,
                value_generator=self.all_weights[0].value_generator,
                сalibration_functions=self.сalibration_functions,
            ),
        )
        population.neuronets.append(self)
        return population.tich(
            dataset=dataset, fertility=fertility, success=success,
            mutability=mutability,
        )

    @classmethod
    def init_from_weights(
        cls, weights: list, structure: list, сalibration_functions=None,
        transmition_function=sig,
    ):
        new_perceptron = cls(
            structure, transmition_function=transmition_function,
        )
        for position, weight in enumerate(new_perceptron.all_weights):
            weight.value = weights[position]
        new_perceptron.сalibration_functions = сalibration_functions
        return new_perceptron

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def all_weights(self) -> list:
        neurons = self.all_neurons
        weights = niccolum_flatten([neuron.weights for neuron in neurons])
        return weights

    @property
    def all_neurons(self) -> list:
        layers = self.layers
        neurons = niccolum_flatten([layer.neurons for layer in layers])
        return neurons
