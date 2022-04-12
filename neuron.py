from utils import generate_sign, generate_uniform, heaviside, softsign


class Weight():
    def __init__(self, value_generator=generate_sign):
        self.value = value_generator()
        self.value_generator = value_generator

    @classmethod
    def init_with_value(cls, value):
        new_weight = cls()
        new_weight.value = value
        return new_weight

    def __repr__(self):
        return f'< Weight: {self.value} >'


class AbstractNeuron:
    def __init__(
        self, inputs_number: int, transmition_function, value_generator,
    ):
        self.transmition_function = transmition_function
        self.weights = list()
        for _weight in range(inputs_number):
            self.weights.append(Weight(value_generator=value_generator))

    def _get_weighted_sum(self, inputs_values: list):
        weighted_values = list()
        for weight_number, weight in enumerate(self.weights):
            weighted_values.append(inputs_values[weight_number] * weight.value)
        return sum(weighted_values)

    def get_output(self, inputs_values: list):
        return self.transmition_function(self._get_weighted_sum(inputs_values))


class InputNeuron(AbstractNeuron):
    def __init__(self, inputs_number: int):
        super().__init__(
            inputs_number=inputs_number,
            transmition_function=heaviside,
            value_generator=generate_uniform,
        )


class StopNeuron(InputNeuron):
    pass


class ContinueNeuron(InputNeuron):
    pass


class InternalLayerNeuron(AbstractNeuron):
    def __init__(self, inputs_number: int):
        super().__init__(
            inputs_number=inputs_number,
            transmition_function=heaviside,
            value_generator=generate_sign,
        )


class OutputNeuron(AbstractNeuron):
    def __init__(self, inputs_number: int):
        super().__init__(
            inputs_number=inputs_number,
            transmition_function=softsign,
            value_generator=generate_uniform,
        )


class NumberNeuron(OutputNeuron):
    pass


class ValueNeuron(OutputNeuron):
    pass


class BiasNeuron(AbstractNeuron):
    def __init__(self):
        super().__init__(
            inputs_number=0,
            transmition_function=None,
            value_generator=None,
        )
        self.inversed = False

    def get_output(self, *args) -> int:
        if self.inversed:
            return -1
        return 1


class ControlCouple:
    def __init__(self, inputs_number):
        self.number = NumberNeuron(inputs_number)
        self.value = ValueNeuron(inputs_number)

    def get_outputs(self, inputs_values: list):
        return (
            self.number.get_output(inputs_values),
            self.value.get_output(inputs_values),
        )
