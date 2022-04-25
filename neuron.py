from utils import (count_arrays_product, generate_sign, generate_uniform,
                   heaviside, softsign)


class Weight:
    def __init__(self, value_generator=generate_sign):
        self.value = value_generator()
        self.value_generator = value_generator

    @classmethod
    def init_with_value(cls, value: float):
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

    @property
    def weights_values(self):
        return [weight.value for weight in self.weights]

    def _get_weighted_sum(self, inputs_values: list):
        return sum(count_arrays_product(self.weights_values, inputs_values))

    def get_output(self, inputs_values: list):
        return self.transmition_function(self._get_weighted_sum(inputs_values))


class InputNeuron(AbstractNeuron):
    def __init__(self, inputs_number: int):
        super().__init__(
            inputs_number=inputs_number,
            transmition_function=heaviside,
            value_generator=generate_uniform,
        )


StopNeuron = type('StopNeuron', (InputNeuron,), dict())
ContinueNeuron = type('ContinueNeuron', (InputNeuron,), dict())


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


NumberNeuron = type('NumberNeuron', (OutputNeuron,), dict())
ValueNeuron = type('ValueNeuron', (OutputNeuron,), dict())


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
    def __init__(self, inputs_number: int):
        self.number = NumberNeuron(inputs_number)
        self.value = ValueNeuron(inputs_number)

    def get_outputs(self, inputs_values: list):
        return (
            self.number.get_output(inputs_values),
            self.value.get_output(inputs_values),
        )
