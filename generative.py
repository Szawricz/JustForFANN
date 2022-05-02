from functools import lru_cache
from string import printable
from time import time

from numpy import mean
from textdistance import jaro_winkler

from neuron import BiasNeuron, ContinueNeuron, StopNeuron
from perceptron import Perceptron
from recurent import RecurentPerceptron

jaro_winkler = lru_cache()(jaro_winkler)


class NonComplimentalRecurent(RecurentPerceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure, control_couples_number)
        inputs_number = self.prelast_outputs_number

        self.stop_neuron = StopNeuron(inputs_number)
        self.continue_neuron = ContinueNeuron(inputs_number)

    def get_outputs(
        self, inputs_values_list, time_limit=None, just_final=False,
    ):
        self._set_biases_default()
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

                    if just_final:
                        resoults = list().append(resoult)
                    else:
                        resoults.append(resoult)
                if is_stop:
                    break
            if resoults:
                inputs_values_list = resoults
                self._set_biases_inversed()
            else:
                is_stop = True
        if just_final:
            return resoult
        return resoults

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + [self.stop_neuron, self.continue_neuron]

    def _set_biases_inversed(self):
        for neuron in self.all_neurons:
            if isinstance(neuron, BiasNeuron):
                neuron.inversed = True
    
    def _set_biases_default(self):
        for neuron in self.all_neurons:
            if isinstance(neuron, BiasNeuron):
                neuron.inversed = False



class JustFinalRecurent(NonComplimentalRecurent):
    def get_outputs(self, inputs_values_list, time_limit=None):
        return super().get_outputs(
            inputs_values_list, time_limit, just_final=True,
        )

    def count_error(self, dataset, time_limit=None) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs, time_limit)
            max_unit_error = self.max_unit_error(case_outputs, real_outputs)
            cases_errors.append(max_unit_error)
        self.error = max(cases_errors)
        return self


class LevelsRecurent(NonComplimentalRecurent):
    def __init__(self, structure, control_couples_number=1, levels_number=512):
        super().__init__(structure, control_couples_number)
        self.essential_attrs['levels_number'] = levels_number
        self.levels_number = levels_number

    def count_error(self, dataset, time_limit=None) -> float:
        cases_errors = list()
        for case_inputs, waited_string in dataset:
            real_string = self.get_outputs(case_inputs, time_limit)
            cases_errors.append(1 - jaro_winkler(waited_string, real_string))
        self.error = mean(cases_errors)
        return self

    def get_outputs(self, inputs_values_list, time_limit=None) -> str:
        raw_outputs = super().get_outputs(inputs_values_list, time_limit)
        numbers_list = [number.pop() for number in raw_outputs]
        return self._numbers_to_level_chars_string(numbers_list)

    def _numbers_to_level_chars_string(self, numbers_list: list) -> str:
        chars_list = list()
        for number in numbers_list:
            char = chr(round((number + 1) / 2 * self.levels_number))
            chars_list.append(char)
        return str().join(chars_list)


class ChatBot(LevelsRecurent):
    def __init__(self, structure, control_couples_number=1, charset=printable):
        super().__init__(structure, control_couples_number)
        self.levels_number = len(charset)

        self.essential_attrs.pop('levels_number')
        self.essential_attrs['charset'] = charset

        self.charset = charset

    def get_outputs(self, input_string, time_limit=None) -> str:
        numbers_list = self._string_to_numbers_list(input_string)
        inputs_list = [[number] for number in numbers_list]
        return super().get_outputs(inputs_list, time_limit)

    @staticmethod
    def _string_to_numbers_list(string: str) -> list:
        return [ord(character) for character in string]

    def _numbers_to_level_chars_string(self, numbers_list: list) -> str:
        chars_list = list()
        for number in numbers_list:
            index = round((number + 1) / 2 * (self.levels_number - 1))
            chars_list.append(self.charset[index])
        return str().join(chars_list)
