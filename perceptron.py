from iteration_utilities import deepflatten

from layer import InputLayer, InternalLayer, OutputLayer
from population import Population
from utils import pickling


@pickling
class Perceptron:
    def __init__(self, structure: list):
        # self.recurent define parrents neuronets needs retiching
        self.recurent = False

        self.structure = structure
        # self.essential_attrs define neuronet structure...
        # ...for a similar neuronet making
        self.essential_attrs = [self.structure]

        self.error = None

        self.layers = list()
        for layer_number, neurons_number in enumerate(self.structure):
            # first layer (index=0) of the structure isn't a true layer,
            # but  the next layer neuron's inputs number.
            if layer_number == 0:
                continue
            elif layer_number == 1:
                layer_type = InputLayer
            elif layer_number == len(self.structure) - 1:
                layer_type = OutputLayer
            else:
                layer_type = InternalLayer

            # a last layer neurons` number consider bias adding
            last_layer_neurons_number = self.structure[layer_number - 1] + 1

            layer = layer_type(last_layer_neurons_number, neurons_number)
            self.layers.append(layer)

    def __repr__(self):
        return f'< Perceptron: {self.structure}> '

    def get_outputs(self, inputs_values):
        prelast_outputs = self._get_prelast_outputs(inputs_values)
        return self.layers[-1].get_outputs(prelast_outputs)

    def _get_prelast_outputs(self, inputs_values):
        inputs_values = list(inputs_values)
        inputs_values.insert(0, 1)  # last layer bias output emitation

        resoults = inputs_values
        for layer in self.layers[:-1]:
            resoults = layer.get_outputs(resoults)

        # save prelast outputs for the reccurent special nurons use them later
        self.prelast_outputs = resoults
        return resoults

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
        # If neuronet is downloaded from file and contents population...
        # ...we'll use this population.
        if hasattr(self, 'population'):
            population = self.population

            # if population is bigger or less then given size 
            population.change_size_to(size)

            # delete population from the neuronet for easify its learning
            del self.population
        else:
            # create population with empty place for neuronet itself
            population = Population(size=size-1, neuronet=self)

            population.neuronets.append(self)

        return population.tich(
            dataset=dataset, mortality=mortality, error=error,
            mutability=mutability, time_limit=time_limit, ann_path=ann_path,
            save_population=save_population,
        )

    def copy_with_new_weights(self, weights: list):
        new_perceptron = self.__class__(*self.essential_attrs)
        for position, weight in enumerate(new_perceptron.all_weights):
            weight.value = weights[position]
        return new_perceptron

    def copy_with_random_weights(self):
        return self.__class__(*self.essential_attrs)

    @property
    def prelast_outputs_number(self) -> int:
        return len(self.layers[-2].neurons)

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def all_weights(self) -> list:
        return list(
            deepflatten([neuron.weights for neuron in self.all_neurons]),
        )

    @property
    def all_neurons(self) -> list:
        return list(deepflatten([layer.neurons for layer in self.layers]))
