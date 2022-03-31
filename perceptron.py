from iteration_utilities import deepflatten

from layer import InputLayer, InternalLayer, OutputLayer
from population import Population
from utils import PickleMixin


class Perceptron(PickleMixin):
    def __init__(self, structure: list):
        self.error = None
        self.structure = structure
        self.essential_attrs = [self.structure]
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

    def _get_prelast_outputs(self, inputs_values):
        inputs_values = list(inputs_values)
        inputs_values.insert(0, 1)
        resoults = inputs_values
        for layer in self.layers[:-1]:
            resoults = layer.get_outputs(resoults)
        self.prelast_outputs = resoults
        return resoults

    def get_outputs(self, inputs_values):
        return self.layers[-1].get_outputs(
            self._get_prelast_outputs(inputs_values),
        )

    @staticmethod
    def max_unit_error(case_output, real_output):
        unit_errors = list()
        for waited, real in zip(case_output, real_output):
            unit_errors.append(abs((waited - real) / 2))
        return max(unit_errors)

    def count_error(self, dataset, *args):
        resoults = list()
        for dataset_inputs, dataset_outputs in dataset:
            resoult_outputs = self.get_outputs(dataset_inputs)
            unit_error = self.max_unit_error(dataset_outputs, resoult_outputs)
            resoults.append(unit_error)
        self.error = max(resoults)

    def tich_by_genetic(
        self, dataset: list, size=100, mortality=0.4, error=0.25,
        mutability=0.2, time_limit=None, ann_path=None, save_population=False,
    ) -> object:
        if hasattr(self, 'population'):
            self.population.change_size_to(size)
        else:
            self.population = Population(size=size-1, neuronet=self)
            self.population.neuronets.append(self)
        self.error = None

        resoult = self.population.tich(
            dataset=dataset, mortality=mortality, error=error, 
            mutability=mutability, time_limit=time_limit, ann_path=ann_path,
            save_population=save_population,
        )
        if not save_population:
            del self.population
        return resoult

    @classmethod
    def init_from_weights(cls, weights: list, essential_attrs: list):
        new_perceptron = cls(*essential_attrs)
        for position, weight in enumerate(new_perceptron.all_weights):
            weight.value = weights[position]
        return new_perceptron

    @property
    def prelast_outputs_number(self) -> int:
        return len(self.layers[-2].neurons)

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def all_weights(self) -> list:
        return list(deepflatten([neuron.weights for neuron in self.all_neurons]))

    @property
    def all_neurons(self) -> list:
        return list(deepflatten([layer.neurons for layer in self.layers]))
