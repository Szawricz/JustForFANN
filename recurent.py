from time import time

from neuron import ContinueNeuron, ControlCouple, StopNeuron
from perceptron import Perceptron


class RecurentPerceptron(Perceptron):
    def __init__(self, structure, control_couples_number=1, just_final=False):
        super().__init__(structure)
        inputs_number = len(self.layers[-2].neurons)
        self.essential_attrs.extend([control_couples_number, just_final])
        self.just_final = just_final
        self.control_couples = list()
        for _couple in range(control_couples_number):
            self.control_couples.append(ControlCouple(inputs_number))
        self.stop_neuron = StopNeuron(inputs_number)
        self.continue_neuron = ContinueNeuron(inputs_number)


    def __repr__(self):
        return f'< RecurentPerceptron: {self.structure}>'

    def _change_weights(self, prelast_outputs):
        for couple in self.control_couples:
            number, value = couple.get_outputs(prelast_outputs)
            index = round(((number + 1) / 2) * (self.weights_number - 1))
            self.all_weights[index].value = value

    def get_outputs(self, inputs_values_list, time_limit=25):
        start_time = time()
        resoults = list()
        for inputs_values in inputs_values_list:
            while True:
                prelast_outputs = self._get_prelast_outputs(inputs_values)
                resoult = super().get_outputs(inputs_values)
                is_stop = self.stop_neuron.get_output(prelast_outputs)
                is_continue = self.continue_neuron.get_output(prelast_outputs)
                self._change_weights(prelast_outputs)
                if is_stop or is_continue:
                    break
                if self.just_final:
                    resoults = resoult
                else:
                    resoults.append(resoult)
                if time() - start_time > time_limit:
                    is_stop = True
                    break
            if is_stop:
                break
            if is_continue:
                continue
        return resoults


    def count_error(self, dataset):
        pass

    @classmethod
    def init_from_weights(
        cls, weights: list, structure: list,
        control_couples_number: int, just_final: bool,
    ):
        new_perceptron = cls(structure, control_couples_number, just_final)
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
        return super().all_neurons + self.control_neurons + [
            self.stop_neuron, self.continue_neuron,
        ]
