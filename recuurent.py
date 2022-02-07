from neuron import ControlCouple
from perceptron import Perceptron


class RecurentPerceptron(Perceptron):
    def __init__(self, structure, control_couples_number: int):
        super().__init__(structure)
        inputs_number = len(self.layers[-2].neurons)
        self.essential_attrs.append(control_couples_number)
        self.control_couples = list()
        for _couple in range(control_couples_number):
            self.control_couples.append(ControlCouple(inputs_number))

    def __repr__(self):
        return f'< RecurentPerceptron: {self.structure}>'

    def _change_weights(self, inputs_values):
        prelast_outputs = self._get_prelast_outputs(inputs_values)
        for couple in self.control_couples:
            number, value = couple.get_outputs(prelast_outputs)
            index = round(((number + 1) / 2) * (self.weights_number - 1))
            self.all_weights[index].value = value

    def get_outputs(self, inputs_values_list, just_final=False):
        resoults = list()
        for inputs_values in inputs_values_list:
            resoult = super().get_outputs(inputs_values)
            self._change_weights(inputs_values)
            if just_final and just_final:
                resoults = resoult
            else:
                resoults.append(resoult)
        return resoults

    @classmethod
    def init_from_weights(
        cls, weights: list, structure: list, control_couples_number: int,
    ):
        new_perceptron = cls(structure, control_couples_number)
        for position, weight in enumerate(new_perceptron.all_weights):
            weight.value = weights[position]
        return new_perceptron

    @property
    def control_neurons(self) -> list:
        control_neurons = list()
        for couple in self.control_couples:
            control_neurons.extend([couple.number, couple.value])
        return control_neurons

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + self.control_neurons
