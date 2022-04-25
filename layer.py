from neuron import BiasNeuron, InputNeuron, InternalLayerNeuron, OutputNeuron


class AbstractLayer:
    def __init__(
        self, neuron_inputs_number: int, neurons_number: int,
        neuron_type, with_bias=True,
    ):
        self.neurons = list()
        if with_bias:
            self.neurons.append(BiasNeuron())
        for _neuron_number in range(neurons_number):
            self.neurons.append(neuron_type(neuron_inputs_number)) 

    def get_outputs(self, inputs_values: list) -> list:
        return [neuron.get_output(inputs_values) for neuron in self.neurons]


class InputLayer(AbstractLayer):
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        super().__init__(neuron_inputs_number, neurons_number, InputNeuron)


class InternalLayer(AbstractLayer):
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        super().__init__(
            neuron_inputs_number, neurons_number, InternalLayerNeuron,
        )


class OutputLayer(AbstractLayer):
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        super().__init__(
            neuron_inputs_number, neurons_number, OutputNeuron,
            with_bias=False,
        )
