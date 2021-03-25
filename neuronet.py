"""Only neuronet module."""

from functools import lru_cache
from random import uniform

from numpy import array, exp


def layers_from_weights(weights: dict):
    """Restore neuronet structure from weights.

    Args:
        weights (dict): weigts of ann

    Returns:
        list: neuronet layers structure
    """
    matrix = []  # Matrix of weights names
    layers = []  # Layers of neuronet
    for key in list(weights.keys()):  # Making matrix of weights names
        # Take 3 'numbers' from weight name
        the_three = key.split(sep='w_')[1].split(sep='_')
        # Make numbers from number character
        for position, from_three in enumerate(the_three):
            the_three[position] = int(from_three)
        matrix.append(the_three)
    # Count the layers neurons excluding last layer
    num = 0
    for _layer, neuron, _peview_lay_neu in matrix:
        if neuron == 1:
            num += 1
        elif neuron != 1 and num != 0:
            layers.append(num - 1)
            num = 0
    layers.append(matrix[-1][1])  # Add the lastlayer neurons number
    return layers


@lru_cache()
def sig(neuro_sum: float):
    """Return the sigmoid function result.

    Args:
        neuro_sum: a weighted sum of neurons

    Returns:
        The return value in float type

    """
    return 2 / (1 + exp(-neuro_sum)) - 1


def neuron_value(arr1, arr2):
    """Return neuron output.

    Args:
        arr1: first list of numbers - results of previous layer
        arr2: second list of numbers - values of weights

    Returns:
        Value in float type

    """
    return sig(sum(array(arr1)*array(arr2)))


class Neuronet(object):
    """Class of ANN."""

    layers = []
    weights = {}

    def __init__(self, layers: list, weights: dict):
        """Init a neuronet with a structure.

        Args:
            layers: a layers sizes list
            weights: weigts of ann connects

        """
        self.layers = layers
        self.weights = weights

    @classmethod
    def from_layers(cls, layers: list):
        """Make neuronet from layers structure.

        Args:
            layers (list): neuronet layers structure

        Returns:
            Neuronet: the neuronnet with random weights
        """
        cls.layers = layers
        for lay_num, lay_size in enumerate(layers):
            if lay_num == 0:  # On the first layer we got not weights
                continue
            for neu_num in range(1, lay_size + 1):
                for neu_num_last_lay in range(layers[lay_num - 1] + 1):
                    cls.weights['w_{lnum}_{nnum}_{nnumll}'.format(
                        lnum=str(lay_num),
                        nnum=str(neu_num),
                        nnumll=str(neu_num_last_lay),
                        )] = uniform(-1, 1)
        return Neuronet(cls.layers, cls.weights)

    @classmethod
    def from_weights(cls, weights: dict, layers=None):
        """Make neuronet from the weights values.

        Args:
            weights (dict): Weights of neuronet
            layers (list): Optional argument with neuronet structure

        Returns:
            Neuronet: the neuronet with the weights values
        """
        cls.weights = weights
        if layers != None:
            cls.layers = layers
        cls.layers = layers_from_weights(weights)
        return Neuronet(cls.layers, cls.weights)


    def forvard_propogation(self, inputs: list):
        """Return the result of ANN working.

        Args:
            inputs: a values tuple for forvard propogation

        Returns:
            The result of forward propogation in list type

        """
        layer_weights = list(self.weights.values())  # List from genom dict
        for lay_num, lay_size in enumerate(self.layers):
            if lay_num == 0:
                layer_result = inputs
                layer_result.insert(0, 1)  # Insert bias value
                continue
            prelayer_neu_num = self.layers[lay_num-1] + 1  # "+1" bias counting
            operating_result = layer_result
            layer_result = []
            for _neu_num in range(lay_size):
                elementar_weights = layer_weights[:prelayer_neu_num]
                layer_weights = layer_weights[prelayer_neu_num:]
                layer_result.append(
                    neuron_value(operating_result, elementar_weights),
                    )
            if lay_num + 1 != len(self.layers):
                layer_result.insert(0, 1)
        return layer_result
