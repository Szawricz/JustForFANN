from unpacking_flatten_lists.funcs import niccolum_flatten
from population import Population
from utils import generate_uniform, sig


class Weight():
    def __init__(self, value_generator=generate_uniform):
        self.value = value_generator()
        self.value_generator = value_generator

    @classmethod
    def init_with_value(cls, value):
        new_weight = cls()
        new_weight.value = value
        return new_weight

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
    def __init__(self, value_generator):
        super().__init__(
            inputs_number=0,
            transmition_function=None,
            value_generator=value_generator,
        )

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
        value_generator=generate_uniform, сalibration_functions=None,
    ):
        self.сalibration_functions = сalibration_functions
        self.structure = structure
        interstructure = [neurons_number + 1 for neurons_number in structure]
        self.layers = list()
        for layer_number, neurons_number in enumerate(interstructure):
            if layer_number == 0:
                continue
            self.layers.append(
                Layer(
                    neurons_number=neurons_number,
                    last_layer_neurons_number=interstructure[layer_number - 1],
                    transmition_function=transmition_function,
                    value_generator=value_generator,
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
        uncalibrated_resoults = resoults[1:]
        if self.сalibration_functions:
            calibrated_resoults = list()
            for position, output in enumerate(uncalibrated_resoults):
                calibrated_resoults.append(
                    self.сalibration_functions[position](output)
                )
            return calibrated_resoults
        return uncalibrated_resoults

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
