from neuron import BiasNeuron, InputNeuron, InternalLayerNeuron, OutputNeuron


class AbstractLayer:
    def __init__(
        self, last_layer_neurons_number: int,
        neurons_number: int, neuron_type, with_bias=True,
    ):
        if with_bias:
            start_neurons = [BiasNeuron()]
        else:
            start_neurons = list()
        self.neurons = start_neurons
        for _neuron_number in range(neurons_number - with_bias):
            neuron = neuron_type(inputs_number=last_layer_neurons_number)
            self.neurons.append(neuron)

    def get_outputs(self, inputs_values: list) -> list:
        return [neuron.get_output(inputs_values) for neuron in self.neurons]


class InputLayer(AbstractLayer):
    def __init__(self, last_layer_neurons_number: int, neurons_number: int):
        super().__init__(
            last_layer_neurons_number, neurons_number, InputNeuron,
        )


class InternalLayer(AbstractLayer):
    def __init__(self, last_layer_neurons_number: int, neurons_number: int):
        super().__init__(
            last_layer_neurons_number, neurons_number, InternalLayerNeuron,
        )


class OutputLayer(AbstractLayer):
    def __init__(self, last_layer_neurons_number: int, neurons_number: int):
        super().__init__(
            last_layer_neurons_number, neurons_number,
            OutputNeuron, with_bias=False,
        )
