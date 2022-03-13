from functools import lru_cache
from time import time

from numpy import mean
from textdistance import jaro_winkler

from neuron import ContinueNeuron, ControlCouple, StopNeuron
from perceptron import Perceptron


jaro_winkler = lru_cache()(jaro_winkler)


class RecurentPerceptron(Perceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure)
        self.essential_attrs.append(control_couples_number)
        self.control_couples = list()
        for _couple in range(control_couples_number):
            self.control_couples.append(
                ControlCouple(self.prelast_outputs_number),
            )

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

    @property
    def control_neurons(self) -> list:
        control_neurons = list()
        for couple in self.control_couples:
            control_neurons.extend([couple.number, couple.value])
        return control_neurons

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + self.control_neurons


class NonComplimentalRecurent(RecurentPerceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure, control_couples_number)
        inputs_number = self.prelast_outputs_number
        self.stop_neuron = StopNeuron(inputs_number)
        self.continue_neuron = ContinueNeuron(inputs_number)

    def get_outputs(self, inputs_values_list, time_limit=None):
        start_time = time()
        is_stop = False
        while not is_stop:
            resoults = list()
            for inputs_values in inputs_values_list:
                while not is_stop:
                    resoult = Perceptron.get_outputs(self, inputs_values)
                    is_stop = self.stop_neuron\
                        .get_output(self.prelast_outputs)
                    is_continue = self.continue_neuron\
                        .get_output(self.prelast_outputs)
                    self._change_weights()
                    if time_limit and ((time() - start_time) > time_limit):
                        is_stop = True
                    if is_stop or is_continue:
                        break
                    resoults.append(resoult)
                if is_stop:
                    break
            if resoults:
                inputs_values_list = resoults
            else:
                is_stop = True
        return resoults

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + [self.stop_neuron, self.continue_neuron]


class JustFinalRecurent(NonComplimentalRecurent):
    def get_outputs(self, inputs_values_list, time_limit=None):
        return super().get_outputs(inputs_values_list, time_limit).pop()

    def count_error(self, dataset) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs)
            max_unit_error = self.max_unit_error(case_outputs, real_outputs)
            cases_errors.append(max_unit_error)
        self.error = max(cases_errors)


class LevelsRecurent(NonComplimentalRecurent):
    def __init__(self, structure, control_couples_number=1, levels_number=512):
        super().__init__(structure, control_couples_number)
        self.essential_attrs.append(levels_number)
        self.levels_number = levels_number

    def get_outputs(self, inputs_values_list, time_limit=None) -> str:
        raw_outputs = super().get_outputs(inputs_values_list, time_limit)
        numbers_list = [number.pop() for number in raw_outputs]
        return self._numbers_to_level_chars_string(numbers_list)

    def _numbers_to_level_chars_string(self, numbers_list: list) -> str:
        chars_list = list()
        for number in numbers_list:
            chars_list.append(
                chr(round((number + 1) / 2 * self.levels_number)),
            )
        return str().join(chars_list)

    def count_error(self, dataset, time_limit=None) -> float:
        cases_errors = list()
        for case_inputs, waited_string in dataset:
            real_string = self.get_outputs(case_inputs, time_limit)
            cases_errors.append(1 - jaro_winkler(waited_string, real_string))
        self.error = mean(cases_errors)
