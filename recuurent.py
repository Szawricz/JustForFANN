from neuron import ControlCouple, StopNeuron
from perceptron import Perceptron


class RecurentPerceptron(Perceptron):
    def __init__(
        self, structure, control_couples_number=1, with_stop_neuron=False,
    ):
        super().__init__(structure)
        inputs_number = len(self.layers[-2].neurons)
        self.essential_attrs.extend([control_couples_number, with_stop_neuron])
        self.control_couples = list()
        for _couple in range(control_couples_number):
            self.control_couples.append(ControlCouple(inputs_number))
        self.stop_neuron = None
        if with_stop_neuron:
            self.stop_neuron = StopNeuron(inputs_number)


    def __repr__(self):
        return f'< RecurentPerceptron: {self.structure}>'

    def _change_weights(self, prelast_outputs):
        for couple in self.control_couples:
            number, value = couple.get_outputs(prelast_outputs)
            index = round(((number + 1) / 2) * (self.weights_number - 1))
            self.all_weights[index].value = value

    def get_outputs(self, inputs_values_list, just_final=False):
        resoults = list()
        for iteration, inputs_values in enumerate(inputs_values_list, start=1):
            resoult = super().get_outputs(inputs_values)
            prelast_outputs = self._get_prelast_outputs(inputs_values)
            self._change_weights(inputs_values)
            if not just_final:
                resoults.append(resoult)
            elif iteration == len(inputs_values_list):
                resoults = resoult
            if self.stop_neuron.get_output(prelast_outputs):
                break
        return resoults

    @classmethod
    def init_from_weights(
        cls, weights: list, structure: list,
        control_couples_number: int, with_stop_neuron: bool,
    ):
        new_perceptron = cls(
            structure, control_couples_number, with_stop_neuron,
        )
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
        return super().all_neurons + self.control_neurons + self.stop_neuron
