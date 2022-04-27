from time import time

from neuron import ControlCouple
from perceptron import Perceptron


class RecurentPerceptron(Perceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure)
        self.essential_attrs['control_couples_number'] = control_couples_number
        self.recurent = True

        self.control_couples = list()
        for _couple in range(control_couples_number):
            couple = ControlCouple(self.prelast_outputs_number)
            self.control_couples.append(couple)

    def _change_weights(self):
        for couple in self.control_couples:
            number, value = couple.get_outputs(self.prelast_outputs)
            index = round((number + 1) / 2 * (self.weights_number - 1))
            self.all_weights[index].value = value

    def get_outputs(self, inputs_values_list, time_limit=None):
        start_time = time()
        resoults = list()
        for inputs_values in inputs_values_list:
            resoult = super().get_outputs(inputs_values)
            self._change_weights()
            resoults.append(resoult)
            if time_limit and time() - start_time > time_limit:
                break
        return resoults

    def count_error(self, dataset) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs)
            case_errors = list()
            for case_output, real_output in zip(case_outputs, real_outputs):
                max_unit_error = self.max_unit_error(case_output, real_output)
                case_errors.append(max_unit_error)
            cases_errors.append(max(case_errors))
        self.error = max(cases_errors)
        return self

    @property
    def control_neurons(self) -> list:
        control_neurons = list()
        for couple in self.control_couples:
            control_neurons.extend([couple.number, couple.value])
        return control_neurons

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + self.control_neurons
