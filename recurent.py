from time import time

from textdistance import jaro_winkler

from neuron import ContinueNeuron, ControlCouple, StopNeuron
from perceptron import Perceptron
from utils import make_simple_structure


class RecurentPerceptron(Perceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure)
        inputs_number = len(self.layers[-2].neurons)
        self.essential_attrs.append(control_couples_number)
        self.control_couples = list()
        for _couple in range(control_couples_number):
            self.control_couples.append(ControlCouple(inputs_number))

    def _change_weights(self, prelast_outputs):
        for couple in self.control_couples:
            number, value = couple.get_outputs(prelast_outputs)
            index = round(((number + 1) / 2) * (self.weights_number - 1))
            self.all_weights[index].value = value

    def get_outputs(self, inputs_values_list, time_limit=25):
        start_time = time()
        resoults = list()
        for inputs_values in inputs_values_list:
            prelast_outputs = self._get_prelast_outputs(inputs_values)
            resoult = super().get_outputs(inputs_values)
            self._change_weights(prelast_outputs)
            resoults.append(resoult)
            if time() - start_time > time_limit:
                break
        return resoults

    def count_error(self, dataset) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs)
            case_errors = list()
            for case_output, real_output in list(
                zip(case_outputs, real_outputs),
            ):
                max_unit_error = self.max_unit_error(case_output, real_output)
                case_errors.append(max_unit_error)
            cases_errors.append(max(case_errors))
        self.error = max(cases_errors)

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


class NonComplimentalRecurent(RecurentPerceptron):
    def __init__(self, structure, control_couples_number=1):
        super().__init__(structure, control_couples_number)
        inputs_number = len(self.layers[-2].neurons)
        self.stop_neuron = StopNeuron(inputs_number)
        self.continue_neuron = ContinueNeuron(inputs_number)

    def get_outputs(self, inputs_values_list, time_limit=25):
        start_time = time()
        resoults = list()
        for inputs_values in inputs_values_list:
            while True:
                prelast_outputs = self._get_prelast_outputs(inputs_values)
                resoult = Perceptron.get_outputs(self, inputs_values)
                is_stop = self.stop_neuron.get_output(prelast_outputs)
                is_continue = self.continue_neuron.get_output(prelast_outputs)
                self._change_weights(prelast_outputs)
                if is_stop or is_continue:
                    break
                resoults.append(resoult)
                if time() - start_time > time_limit:
                    is_stop = True
                    break
            if is_stop:
                break
            if is_continue:
                continue
        return resoults

    @staticmethod
    def _string_to_numbers_list(string: str) -> list:
        return [ord(character) for character in string]

    @staticmethod
    def _numbers_list_to_string(numbers_list: list, unicode_max=1279) -> str:
        chars_list = list()
        for number in numbers_list:
            chars_list.append(chr(round((number + 1) / 2 * unicode_max)))
        return str().join(chars_list)

    def get_answer(self, request: str) -> str:
        numbers_list = self._string_to_numbers_list(request)
        inputs_list = [[number] for number in numbers_list]
        outputs_list = self.get_outputs(inputs_list)
        numbers_list = [number.pop() for number in outputs_list]
        return self._numbers_list_to_string(numbers_list)

    def count_error(self, dataset) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs)
            cases_errors.append(
                jaro_winkler(
                    self._numbers_list_to_string(case_outputs),
                    self._numbers_list_to_string(real_outputs),
                ),
            )
        self.error = max(cases_errors)

    @property
    def all_neurons(self) -> list:
        return super().all_neurons + [self.stop_neuron, self.continue_neuron]


class JustFinalRecurent(NonComplimentalRecurent):
    def get_outputs(self, inputs_values_list, time_limit=25):
        return super().get_outputs(inputs_values_list, time_limit).pop()

    def count_error(self, dataset) -> float:
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            real_outputs = self.get_outputs(case_inputs)
            max_unit_error = self.max_unit_error(case_outputs, real_outputs)
            cases_errors.append(max_unit_error)
        self.error = (cases_errors)




if __name__ == '__main__':
    def chat_with_bot():
        chat_bot = NonComplimentalRecurent(
            structure=make_simple_structure(
                inputs_number=1,
                intermediate_layers_number=7,
                intermediate_layers_neurons_number=100,
                outputs_number=1,
            ),
            control_couples_number=50,
        )
        while True:
            answer = chat_bot.get_answer(input('You: '))
            print(f'\nBot: {answer}\n')

    chat_with_bot()
